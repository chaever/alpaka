///TODO Copyright, License and other stuff here

#pragma once

#include <experimental/simd>

namespace alpaka::lockstep
{
    namespace simdBackendTags{

        class ScalarSimdTag{};
        class ArrayOf4Tag{};
        class StdSimdTag{};
        class StdSimdFourTimesTag{};

    }

    template<typename T_Elem, typename T_Simd>
    struct SimdInterface;

    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem>
    struct SimdInterface<T_Elem, simdBackendTags::ScalarSimdTag>{
        using Elem_t = T_Elem;
        using Pack_t = T_Elem;//special case laneCount=1 : elements are packs

        inline static constexpr std::size_t laneCount = 1u;

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const Elem_t* const mem) -> Pack_t
        {
            return *mem;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, Elem_t* const mem)
        {
            *mem = t;
        }
    };

    //Specialization using std::experimental::simd
    template<typename T_Elem>
    struct SimdInterface<T_Elem, simdBackendTags::StdSimdTag>{
        using Elem_t = T_Elem;
        using Pack_t = std::experimental::simd<T_Elem, std::experimental::simd_abi::native<T_Elem>>;

        inline static constexpr std::size_t laneCount = Pack_t::size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(Elem_t const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, Elem_t * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }
    };

    //std::experimental::simd, but 4 at a time
    template<typename T_Elem>
    struct SimdInterface<T_Elem, simdBackendTags::StdSimdFourTimesTag>{
        using Elem_t = T_Elem;
        static constexpr auto sizeOfNativePack = std::experimental::simd<T_Elem, std::experimental::simd_abi::native<T_Elem> >::size();
        using Pack_t =std::experimental::simd<T_Elem, std::experimental::simd_abi::fixed_size<sizeOfNativePack*4> >;

        inline static constexpr std::size_t laneCount = Pack_t::size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(Elem_t const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, Elem_t * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }
    };

    template<typename T_Elem>
    using StdArrayWith4Elems = std::array<T_Elem, alpaka::core::vectorization::GetVectorizationSizeElems<T_Elem>::value>;

    //Specialization using 4 values in an array
    template<typename T_Elem>
    struct SimdInterface<T_Elem, simdBackendTags::ArrayOf4Tag>{
        using Elem_t = T_Elem;
        using Pack_t = StdArrayWith4Elems<T_Elem>;

        inline static constexpr std::size_t laneCount = Pack_t{}.size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const Elem_t* const mem) -> Pack_t
        {
            Pack_t tmp;
            for(auto i=0u; i<laneCount; ++i)
                tmp[i] = mem[i];
            return tmp;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t pack, Elem_t* const mem)
        {
            for(auto i=0u; i<laneCount; ++i)
                mem[i] = pack[i];
        }

        template<typename T_Other>
        ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC decltype(auto) operator+(T_Other const & other){
            Pack_t tmp;
            for(auto i=0u; i<laneCount; ++i){
                tmp[i]=*this[i] + other[i];
            }
            return tmp;
        }
    };

    //provides Information about the pack framework the user selected via CMake
    namespace simdBackendTags{
#if defined COMPILE_OPTION_FROM_CMAKE_1
        using SelectedSimdBackendTag = simdBackendTags::StdSimdTag;
#elif defined COMPILE_OPTION_FROM_CMAKE_2
        using SelectedSimdBackendTag = simdBackendTags::ArrayOf4Tag;
#elif 1 || defined COMPILE_OPTION_FROM_CMAKE_3
        using SelectedSimdBackendTag = simdBackendTags::StdSimdFourTimesTag;
#else
        using SelectedSimdBackendTag = simdBackendTags::ScalarSimdTag;
#endif
    }

    //conforms to the SimdInterface class above
    template<typename T_Type>
    using SimdInterface_t = SimdInterface<T_Type, simdBackendTags::SelectedSimdBackendTag>;

    //lane count for any type T, using the selected SIMD backend
    template<typename T_Type>
    static constexpr size_t laneCount = SimdInterface_t<T_Type>::laneCount;

    template<typename T_Type>
    using Pack_t = typename SimdInterface_t<T_Type>::Pack_t;

    template<uint32_t T_offset = 0u>
    class ScalarLookupIndex{
        std::size_t m_idx;
    public:

        static constexpr auto offset = T_offset;

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE ScalarLookupIndex (const std::size_t idx) : m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }
    };

    class SimdLookupIndex{
        std::size_t m_idx;
    public:

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE SimdLookupIndex (const std::size_t idx) : m_idx(idx){}

        //allow conversion to flat number, but not implicitly
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit constexpr operator uint32_t() const
        {
            return m_idx;
        }
    };
} // namespace alpaka::lockstep
