///TODO Copyright, License and other stuff here

#pragma once

//on the CPU, we should have std::simd.
#if defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#define ALPAKA_USE_STD_SIMD 1

#include <StdSimdOperators.hpp>

#endif

namespace alpaka::lockstep
{
    //provides Information about the pack framework the user selected via CMake
    namespace simdBackendTags{

        class ScalarSimdTag{};
        template<uint32_t T_simdMult>
        class StdSimdNTimesTag{};

#if 1 && ALPAKA_USE_STD_SIMD
        using SelectedSimdBackendTag = simdBackendTags::StdSimdNTimesTag<1>;
#else
        //GPU must use scalar simd packs
        using SelectedSimdBackendTag = simdBackendTags::ScalarSimdTag;
#endif
    } // namespace simdBackendTags

    template<typename T_Elem, typename T_SizeIndicator, typename T_SimdBackend = simdBackendTags::SelectedSimdBackendTag>
    struct PackWrapper;

    //lane count for any type T, using the selected SIMD backend
    template<typename T_SizeIndicator>
    static constexpr size_t laneCount_v = PackWrapper<T_SizeIndicator, T_SizeIndicator>::laneCount;

    template<typename T_Type, typename T_SizeIndicator>
    using Pack_t = PackWrapper<T_Type, T_SizeIndicator>;

    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem, typename T_SizeIndicator>
    struct PackWrapper<T_Elem, T_SizeIndicator, simdBackendTags::ScalarSimdTag>{
        T_Elem packContent;

        //broadcast
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(T_Elem const& elem):packContent(elem){}

        //conversion from other packs
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> pack):packContent(static_cast<T_Elem>(pack.packContent)){}
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> const& pack):packContent(static_cast<T_Elem>(pack.packContent)){}
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> && pack):packContent(static_cast<T_Elem>(pack.packContent)){}

        using Elem_t = T_Elem;
        using SizeIndicator_t = T_SizeIndicator;

        inline constexpr std::size_t laneCount = 1u;

        template<T>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T&&) const {
            return packContent;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto loadUnaligned(const T_Elem* const mem)
        {
            return PackWrapper{*mem};
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr void storeUnaligned(const PackWrapper t, T_Elem * const mem)
        {
            *mem = t.packContent;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto& broadcast(PackWrapper const & elem)
        {
            return elem;
        }

        //implicit conversion to the wrapped type
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr operator T_Elem() const {
            return packContent;
        }
    };

#if ALPAKA_USE_STD_SIMD

    namespace trait{

        template<typename T_From, typename T_To, typename T_BackendTag>
        struct simdPackCast;

        //convert pack->pack
        template<typename T_SourceElem, typename T_DestElem, uint32_t T_simdMult>
        struct simdPackCast<PackWrapper<T_SourceElem, T_SourceElem>, PackWrapper<T_DestElem, T_DestElem>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceElem> == laneCount_v<T_DestElem>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr get(PackWrapper<T_SourceElem, T_SourceElem> const& pack){
                using Abi_t = typename PackWrapper<T_DestElem, T_DestElem>::abi_multiplied_t;
                return PackWrapper<T_DestElem, T_DestElem>{std::experimental::static_simd_cast<T_DestElem, T_SourceElem, Abi_t>(pack.packContent)};
            }
        };

        //convert pack->mask
        template<typename T_SourceElem, typename T_DestSizeInd, uint32_t T_simdMult>
        struct simdPackCast<PackWrapper<T_SourceElem, T_SourceElem>, PackWrapper<bool, T_DestSizeInd>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceElem> == laneCount_v<T_DestSizeInd>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr get(PackWrapper<T_SourceElem, T_SourceElem> const& pack){
                return PackWrapper<bool, T_DestSizeInd>{pack.packContent != 0};
            }
        };

        //convert mask->pack
        template<typename T_SourceSizeInd, typename T_DestElem, uint32_t T_simdMult>
        struct simdPackCast<PackWrapper<bool, T_SourceSizeInd>, PackWrapper<T_DestElem, T_DestElem>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceSizeInd> == laneCount_v<T_DestElem>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr get(PackWrapper<bool, T_SourceSizeInd> const& mask){

                static_assert(std::is_arithmetic_v<T_DestElem>);
                using targetPack_t = typename PackWrapper<T_DestElem, T_DestElem>::multiplied_Pack_t;
                targetPack_t tmp(0);
                std::experimental::where(mask.packContent, tmp) = targetPack_t(1);
                return PackWrapper<T_DestElem, T_DestElem>{tmp};
            }
        };

        //convert mask->mask
        template<typename T_SourceSizeInd, typename T_DestSizeInd, uint32_t T_simdMult>
        struct simdPackCast<PackWrapper<bool, T_SourceSizeInd>, PackWrapper<bool, T_DestSizeInd>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceSizeInd> == laneCount_v<T_DestSizeInd>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr get(PackWrapper<bool, T_SourceSizeInd> const& mask){
                using targetMask_t = typename PackWrapper<bool, T_DestSizeInd>::multiplied_Pack_t;
                return PackWrapper<bool, T_DestSizeInd>{targetMask_t{mask.packContent}};
            }
        };
    } // namespace trait

    //std::experimental::simd, but N at a time
    template<typename T_Elem, typename T_SizeIndicator, uint32_t T_simdMult>
    struct PackWrapper<T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
        inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;

        //make sure that only bool can have T_Elem != T_SizeIndicator
        static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
        static_assert(!std::is_same_v<bool, T_SizeIndicator>);

        using abi_native_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        inline static constexpr auto sizeOfNativePack = std::experimental::simd<T_SizeIndicator, abi_native_t>::size();
        using abi_multiplied_t = std::experimental::simd_abi::fixed_size<sizeOfNativePack*T_simdMult>;
        using multiplied_Pack_t = std::conditional_t<packIsMask,
        std::experimental::simd_mask<T_SizeIndicator, abi_multiplied_t>,
        std::experimental::simd<T_Elem, abi_multiplied_t>>;

        ///TODO inherit from multiplied_Pack_t (would also inherit all operators that std::simd packs have)
        multiplied_Pack_t packContent;

        //broadcast
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(multiplied_Pack_t const& elem):packContent(elem){}

        //conversion from other packs
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> pack):
        packContent(trait::simdPackCast<T_SourceElem, T_SourceSizeInd, T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult> >::get(pack).packContent){}
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> const& pack):
        packContent(trait::simdPackCast<T_SourceElem, T_SourceSizeInd, T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult> >::get(pack).packContent){}
        template<typename T_SourceElem, typename T_SourceSizeInd>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(PackWrapper<T_SourceElem, T_SourceSizeInd> && pack):
        packContent(trait::simdPackCast<T_SourceElem, T_SourceSizeInd, T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult> >::get(pack).packContent){}

        using Elem_t = T_Elem;
        using SizeIndicator_t = T_SizeIndicator;

        inline constexpr std::size_t laneCount = multiplied_Pack_t::size();

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](uint32_t const i) const {
            return packContent[i];
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto loadUnaligned(T_Elem const * const mem)
        {
            return PackWrapper{multiplied_Pack_t{mem, std::experimental::element_aligned}};
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr void storeUnaligned(const PackWrapper t, T_Elem * const mem)
        {
            t.packContent.copy_to(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto broadcast(T_Elem const & elem)
        {
            return PackWrapper{multiplied_Pack_t{elem}};
        }
    };

#endif

    ///TODO operator definitions
/*
    template<typename T_Left, typename T_Right>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator+(PackWrapper<T_LeftElem, T_LeftSizeInd> const& left, PackWrapper<T_RightElem, T_RightSizeInd> const& right){

        using result_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>()+std::declval<typename std::decay_t<decltype(right)>::Elem_t>());



        //what about result_t == bool

        PackWrapper<result_t, result_t>









        return left.packContent + right.packContent;
    }
*/


    template<typename T_Foreach, uint32_t T_offset = 0u>
    class ScalarLookupIndex{
        std::size_t m_idx;
    public:

        static constexpr auto offset = T_offset;
        T_Foreach const& m_forEach;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ScalarLookupIndex(T_Foreach const& forEach, const std::size_t idx): m_forEach(forEach), m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }
    };

    template<typename T_Foreach>
    class SimdLookupIndex{
        std::size_t m_idx;
    public:

        T_Foreach const& m_forEach;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr SimdLookupIndex (T_Foreach const& forEach, const std::size_t idx) : m_forEach(forEach), m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }

#undef ALPAKA_USE_STD_SIMD
    };
} // namespace alpaka::lockstep
