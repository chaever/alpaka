///TODO Copyright, License and other stuff here

#pragma once

//on the CPU, we should have std::simd.
#if defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#define ALPAKA_USE_STD_SIMD 1

#include "StdSimdOperators.hpp"

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

    template<typename T_Elem, typename T_SizeIndicator, typename T_SimdBackend>
    struct PackWrapper;

    inline namespace trait{

        template<typename T_Type>
        struct IsPackWrapper{
            static constexpr bool value = false;
        };

        template<typename T_Elem, typename T_SizeIndicator, typename T_SimdBackend>
        struct IsPackWrapper<PackWrapper<T_Elem, T_SizeIndicator, T_SimdBackend>>{
            static constexpr bool value = true;
        };
    }

    template<typename T_Type>
    constexpr bool isPackWrapper_v = trait::IsPackWrapper<T_Type>::value;

    template<typename T_Type, typename T_SizeIndicator>
    using Pack_t = PackWrapper<T_Type, T_SizeIndicator, simdBackendTags::SelectedSimdBackendTag>;

    //lane count for any type T, using the selected SIMD backend
    template<typename T_SizeIndicator>
    static constexpr size_t laneCount_v = Pack_t<T_SizeIndicator, T_SizeIndicator>::laneCount;

    template<typename T_Type, typename T_SizeIndicator>
    using OneElemPack_t = PackWrapper<T_Type, T_SizeIndicator, simdBackendTags::ScalarSimdTag>;

    //3rd template parameter is needed for different types because of https://stackoverflow.com/questions/65870517/understanding-when-a-default-template-parameter-is-a-redefinition-and-when-not-i
    template<typename T_Elem, typename T_SizeIndicator, typename T_ResultIfEnabled>
    using disable_if_PackAndOneElemPackAreEqual = std::enable_if_t<!std::is_same_v<OneElemPack_t<T_Elem, T_SizeIndicator>, Pack_t<T_Elem, T_SizeIndicator> >, T_ResultIfEnabled>;

    //for loading packs from non-contiguous memory
    template<typename T_Lambda>
    struct MemAccessorFunctor{
        T_Lambda const m_lambda;
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr MemAccessorFunctor(T_Lambda const lambda):m_lambda(lambda){}

        constexpr MemAccessorFunctor(MemAccessorFunctor const&) = default;
        constexpr MemAccessorFunctor(MemAccessorFunctor &)      = default;
        constexpr MemAccessorFunctor(MemAccessorFunctor &&)     = default;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[] (const uint32_t i) const{
            return m_lambda(i);
        }
    };

    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem, typename T_SizeIndicator>
    struct PackWrapper<T_Elem, T_SizeIndicator, simdBackendTags::ScalarSimdTag>{
        T_Elem packContent;

        static_assert(!std::is_reference_v<T_Elem>);
        static_assert(!std::is_reference_v<T_SizeIndicator>);

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper() = default;

        //broadcast
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(T_Elem const& elem):packContent(elem){}

        //conversion from other packs
        template<typename T_Elem_, typename T_SizeIndicator_>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(OneElemPack_t<T_Elem_, T_SizeIndicator_> const& pack):packContent(static_cast<T_Elem>(pack.packContent)){}

        template<typename T_Lambda>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(MemAccessorFunctor<T_Lambda> const loader):packContent(loader[0u]){}

        using Elem_t = T_Elem;
        using SizeIndicator_t = T_SizeIndicator;

        static constexpr std::size_t laneCount = 1u;

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper& operator=(PackWrapper const&) = default;

        template<typename T_Lhs>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper operator+=(T_Lhs const& lhs){
            //cast to own type and add that
            return PackWrapper(packContent += PackWrapper(lhs).packContent);
        }

        template<typename T_Lhs>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper operator-=(T_Lhs const& lhs){
            return PackWrapper(packContent -= PackWrapper(lhs).packContent);
        }

        template<typename T>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto& operator[](T&&) {
            return packContent;
        }

        template<typename T>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto const& operator[](T&&) const {
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

        //implicit conversion to the wrapped type
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr operator T_Elem() const {
            return packContent;
        }
    };

#if ALPAKA_USE_STD_SIMD

    inline namespace trait{

        template<typename T_From, typename T_To, typename T_BackendTag>
        struct simdPackCast;

        //convert pack->pack
        template<typename T_SourceElem, typename T_DestElem, uint32_t T_simdMult>
        struct simdPackCast<Pack_t<T_SourceElem, T_SourceElem>, Pack_t<T_DestElem, T_DestElem>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceElem> == laneCount_v<T_DestElem>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto get(Pack_t<T_SourceElem, T_SourceElem> const& pack){
                using Abi_t = typename Pack_t<T_DestElem, T_DestElem>::abi_multiplied_t;
                return Pack_t<T_DestElem, T_DestElem>{std::experimental::static_simd_cast<T_DestElem, T_SourceElem, Abi_t>(pack.packContent)};
            }
        };

        //convert pack->mask
        template<typename T_SourceElem, typename T_DestSizeInd, uint32_t T_simdMult>
        struct simdPackCast<Pack_t<T_SourceElem, T_SourceElem>, Pack_t<bool, T_DestSizeInd>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceElem> == laneCount_v<T_DestSizeInd>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto get(Pack_t<T_SourceElem, T_SourceElem> const& pack){
                return Pack_t<bool, T_DestSizeInd>{pack.packContent != 0};
            }
        };

        //convert mask->pack
        template<typename T_SourceSizeInd, typename T_DestElem, uint32_t T_simdMult>
        struct simdPackCast<Pack_t<bool, T_SourceSizeInd>, Pack_t<T_DestElem, T_DestElem>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceSizeInd> == laneCount_v<T_DestElem>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto get(Pack_t<bool, T_SourceSizeInd> const& mask){

                static_assert(std::is_arithmetic_v<T_DestElem>);
                using targetPack_t = typename Pack_t<T_DestElem, T_DestElem>::multiplied_Pack_t;
                targetPack_t tmp(0);
                std::experimental::where(mask.packContent, tmp) = targetPack_t(1);
                return Pack_t<T_DestElem, T_DestElem>{tmp};
            }
        };

        //convert mask->mask
        template<typename T_SourceSizeInd, typename T_DestSizeInd, uint32_t T_simdMult>
        struct simdPackCast<Pack_t<bool, T_SourceSizeInd>, Pack_t<bool, T_DestSizeInd>, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
            static_assert(laneCount_v<T_SourceSizeInd> == laneCount_v<T_DestSizeInd>);
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto get(Pack_t<bool, T_SourceSizeInd> const& mask){
                using targetMask_t = typename Pack_t<bool, T_DestSizeInd>::multiplied_Pack_t;
                return Pack_t<bool, T_DestSizeInd>{targetMask_t{mask.packContent}};
            }
        };
    } // namespace trait

    //std::experimental::simd, but N at a time
    template<typename T_Elem, typename T_SizeIndicator, uint32_t T_simdMult>
    struct PackWrapper<T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
        inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;

        static_assert(!std::is_reference_v<T_Elem>);
        static_assert(!std::is_reference_v<T_SizeIndicator>);

        //make sure that only bool can have T_Elem != T_SizeIndicator
        static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
        static_assert(!std::is_same_v<bool, T_SizeIndicator>);

        using abi_native_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        inline static constexpr auto sizeOfNativePack = std::experimental::simd<T_SizeIndicator, abi_native_t>::size();
        using abi_multiplied_t = std::experimental::simd_abi::fixed_size<sizeOfNativePack*T_simdMult>;
        using multiplied_Pack_t = std::conditional_t<packIsMask,
        std::experimental::simd_mask<T_SizeIndicator, abi_multiplied_t>,
        std::experimental::simd<T_Elem, abi_multiplied_t>>;

        //not inherited because std::simd's ops are not as universal as we need them to be
        multiplied_Pack_t packContent;

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper() = default;

        //initialization from underlying pack type
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(multiplied_Pack_t const& elem):packContent(elem){}

        //broadcast
        template<typename T, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T>>, int> = 0>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(T const& elem):packContent(static_cast<T_Elem>(elem)){}

        //broadcast
        template<typename T_Elem_, typename T_SizeIndicator_, std::enable_if_t<!std::is_same_v<OneElemPack_t<T_Elem_, T_SizeIndicator_>, Pack_t<T_Elem_, T_SizeIndicator_>>, int> = 0>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(OneElemPack_t<T_Elem_, T_SizeIndicator_> const& pack):packContent(static_cast<T_Elem>(pack.packContent)){}

        //conversion from other packs
        template<typename T_Elem_, typename T_SizeIndicator_>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(Pack_t<T_Elem_, T_SizeIndicator_> const& pack):
        packContent(trait::simdPackCast<Pack_t<T_Elem_, T_SizeIndicator_>, PackWrapper, simdBackendTags::StdSimdNTimesTag<T_simdMult> >::get(pack).packContent){}

        template<typename T_Lambda>
        explicit ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper(MemAccessorFunctor<T_Lambda> const loader):packContent(loader.m_lambda){}

        using Elem_t = T_Elem;
        using SizeIndicator_t = T_SizeIndicator;

        static constexpr std::size_t laneCount = multiplied_Pack_t::size();

        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper& operator=(PackWrapper const&) = default;

        template<typename T_Lhs>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper operator+=(T_Lhs const& lhs){
            //cast to own type and add that
            return PackWrapper(packContent += PackWrapper(lhs).packContent);
        }

        template<typename T_Lhs>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr PackWrapper operator-=(T_Lhs const& lhs){
            return PackWrapper(packContent -= PackWrapper(lhs).packContent);
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](uint32_t const i){

            return packContent[i];
        }

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
    };

#endif

#define XPR_OP_WRAPPER() operator

#define BINARY_READONLY_ARITHMETIC_OP(opName)\
    /*Pack op Pack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*Pack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*nonPack op Pack*/\
    template<typename T_LeftNonPack, typename T_RightElem, typename T_RightSizeInd, std::enable_if_t<!isPackWrapper_v<T_LeftNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (T_LeftNonPack const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*Pack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right.packContent).packContent);\
    }\
    /*OneElemPack op Pack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, char> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left.packContent).packContent opName resultPack_t(right).packContent);\
    }\
    /*OneElemPack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, bool> = false>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using resultPack_t = OneElemPack_t<result_elem_t, result_elem_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*OneElemPack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*nonPack op OneElemPack*/\
    template<typename T_LeftNonPack, typename T_RightElem, typename T_RightSizeInd, std::enable_if_t<!isPackWrapper_v<T_LeftNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (T_LeftNonPack const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }

#define BINARY_READONLY_COMPARISON_OP(opName)\
    /*Pack op Pack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = Pack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right).packContent);\
    }\
    /*Pack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = Pack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right).packContent);\
    }\
    /*nonPack op Pack*/\
    template<typename T_LeftNonPack, typename T_RightElem, typename T_RightSizeInd, std::enable_if_t<!isPackWrapper_v<T_LeftNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (T_LeftNonPack const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = Pack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right).packContent);\
    }\
    /*Pack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = Pack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right.packContent).packContent);\
    }\
    /*OneElemPack op Pack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, char> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = Pack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left.packContent).packContent opName comparePack_t(right).packContent);\
    }\
    /*OneElemPack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, bool> = false>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using resultPack_t = OneElemPack_t<result_elem_t, result_elem_t>;\
        return resultPack_t(left.packContent opName right.packContent);\
    }\
    /*OneElemPack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using compare_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() + std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = OneElemPack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right).packContent);\
    }\
    /*nonPack op OneElemPack*/\
    template<typename T_LeftNonPack, typename T_RightElem, typename T_RightSizeInd, std::enable_if_t<!isPackWrapper_v<T_LeftNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (T_LeftNonPack const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using compare_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() + std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        using comparePack_t = OneElemPack_t<compare_elem_t, result_sizeInd_t>;\
        return resultPack_t(comparePack_t(left).packContent opName comparePack_t(right).packContent);\
    }

#define BINARY_READONLY_SHIFT_OP(opName)\
    /*Pack op Pack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, Pack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName resultPack_t(right).packContent);\
    }\
    /*Pack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName right);\
    }\
    /*Pack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (Pack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = Pack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName right.packContent);\
    }\
    /*OneElemPack op OneElemPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightElem, typename T_RightSizeInd, disable_if_PackAndOneElemPackAreEqual<T_LeftElem, T_LeftSizeInd, char> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(left.packContent opName right.packContent);\
    }\
    /*OneElemPack op nonPack*/\
    template<typename T_LeftElem, typename T_LeftSizeInd, typename T_RightNonPack, std::enable_if_t<!isPackWrapper_v<T_RightNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (OneElemPack_t<T_LeftElem, T_LeftSizeInd> const& left, T_RightNonPack const& right){\
        using result_elem_t = decltype(std::declval<typename std::decay_t<decltype(left)>::Elem_t>() opName std::declval<std::decay_t<decltype(right)>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(left)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(resultPack_t(left).packContent opName right);\
    }\
    /*nonPack op OneElemPack*/\
    template<typename T_LeftNonPack, typename T_RightElem, typename T_RightSizeInd, std::enable_if_t<!isPackWrapper_v<T_LeftNonPack>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()opName (T_LeftNonPack const& left, OneElemPack_t<T_RightElem, T_RightSizeInd> const& right){\
        using result_elem_t = decltype(std::declval<std::decay_t<decltype(left)>>() opName std::declval<typename std::decay_t<decltype(right)>::Elem_t>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, typename std::decay_t<decltype(right)>::SizeIndicator_t, result_elem_t>;\
        using resultPack_t = OneElemPack_t<result_elem_t, result_sizeInd_t>;\
        return resultPack_t(left opName resultPack_t(right).packContent);\
    }

#define UNARY_READONLY_OP_PREFIX(opName)\
    template<typename T_Elem, typename T_SizeInd, typename T_SimdBackend>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName(PackWrapper<T_Elem, T_SizeInd, T_SimdBackend> const& pack)\
    {\
        return PackWrapper<T_Elem, T_SizeInd, T_SimdBackend>{opName pack.packContent};\
    }

#define UNARY_READONLY_OP_POSTFIX(opName)\
    template<typename T_Elem, typename T_SizeInd, typename T_SimdBackend>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName(PackWrapper<T_Elem, T_SizeInd, T_SimdBackend> const& pack, [[unused]] int /*neededForPrefixAndPostfixDistinction*/)\
    {\
        return PackWrapper<T_Elem, T_SizeInd, T_SimdBackend>{pack.packContent opName};\
    }

#define UNARY_FREE_FUNCTION(fName)\
    template<typename T_Elem, typename T_SizeInd, typename T_SimdBackend>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto fName(alpaka::lockstep::PackWrapper<T_Elem, T_SizeInd, T_SimdBackend> const& pack){\
        return alpaka::lockstep::PackWrapper<T_Elem, T_SizeInd, T_SimdBackend>(fName(pack.packContent));\
    }

    BINARY_READONLY_ARITHMETIC_OP(+)
    BINARY_READONLY_ARITHMETIC_OP(-)
    BINARY_READONLY_ARITHMETIC_OP(*)
    BINARY_READONLY_ARITHMETIC_OP(/)
    BINARY_READONLY_ARITHMETIC_OP(&)
    BINARY_READONLY_ARITHMETIC_OP(&&)
    BINARY_READONLY_ARITHMETIC_OP(|)
    BINARY_READONLY_ARITHMETIC_OP(||)

    BINARY_READONLY_COMPARISON_OP(>)
    BINARY_READONLY_COMPARISON_OP(<)

    UNARY_READONLY_OP_PREFIX(!)
    UNARY_READONLY_OP_PREFIX(~)

    UNARY_READONLY_OP_POSTFIX(++)
    UNARY_READONLY_OP_POSTFIX(--)

    BINARY_READONLY_SHIFT_OP(>>)
    BINARY_READONLY_SHIFT_OP(<<)

} // namespace alpaka::lockstep

    UNARY_FREE_FUNCTION(std::abs)

namespace alpaka::lockstep
{

#undef UNARY_FREE_FUNCTION
#undef UNARY_READONLY_OP_POSTFIX
#undef UNARY_READONLY_OP_PREFIX
#undef BINARY_READONLY_SHIFT_OP
#undef BINARY_READONLY_COMPARISON_OP
#undef BINARY_READONLY_ARITHMETIC_OP
#undef XPR_OP_WRAPPER

    template<typename T_Foreach, uint32_t T_offset = 0u>
    class ScalarLookupIndex{
        uint32_t m_idx;
    public:

        static constexpr auto offset = T_offset;
        T_Foreach const& m_forEach;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ScalarLookupIndex(T_Foreach const& forEach, const uint32_t idx): m_forEach(forEach), m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }
    };

    template<typename T_Foreach>
    class SimdLookupIndex{
        uint32_t m_idx;
    public:

        T_Foreach const& m_forEach;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr SimdLookupIndex (T_Foreach const& forEach, const uint32_t idx) : m_forEach(forEach), m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }

#undef ALPAKA_USE_STD_SIMD
    };
} // namespace alpaka::lockstep
