///TODO Copyright, License and other stuff here

#pragma once

namespace alpaka::lockstep
{
    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem, typename T_Simd = void>
    struct SimdInterface{
        using Elem_t = T_Elem;
        using Pack_t = T_Elem;//special case laneCount=1 : elements are packs

        inline static constexpr std::size_t laneCount = 1u;

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const Elem_t* mem) -> Pack_t
        {
            return *mem;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(Pack_t t, Elem_t* mem)
        {
            *mem = t;
        }
    };

    //Specialization using std::experimental::simd
    template<typename T_Elem>
    struct SimdInterface<T_Elem, std::experimental::simd<T_Elem>>{
        using Elem_t = T_Elem;
        using Pack_t = std::experimental::simd<T_Elem, std::experimental::simd_abi::native<T_Elem>>;

        inline static constexpr std::size_t laneCount = Pack_t::size;

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const Elem_t* mem) -> Pack_t
        {
            return Pack_t(mem);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(Pack_t t, Elem_t* mem)
        {
            t.copy_to(mem);
        }
    };

    template<typename T_Elem>
    using StdArrayWith4Elems = std::array<T_Elem, alpaka::core::vectorization::GetVectorizationSizeElems<T_Elem>::value>;

    //Specialization using 4 values in an array
    template<typename T_Elem>
    struct SimdInterface<T_Elem, StdArrayWith4Elems<T_Elem>>{
        using Elem_t = T_Elem;
        using Pack_t = StdArrayWith4Elems<T_Elem>;

        inline static constexpr std::size_t laneCount = Pack_t{}.size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const Elem_t* mem) -> Pack_t
        {
            Pack_t tmp;
            for(auto i=0u; i<laneCount; ++i)
                tmp[i] = mem[i];
            return tmp;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(Pack_t pack, Elem_t* mem)
        {
            for(auto i=0u; i<laneCount; ++i)
                mem[i] = pack[i];
        }
    };

    //provides Information about the pack framework the user selected via CMake
    struct selectedSIMDInfo{
#if defined COMPILE_OPTION_FROM_CMAKE_1
        template<typename T>
        using simdNonBool_t = std::experimental::simd<T>;
        //provides as many booleans as simdNonBool_t<T> can hold Ts
        ///TODO decide if needed, can emulate by using other types
        //template<typename T>
        //using simdBool_t = std::experimental::simd_mask<T>;
#elif defined COMPILE_OPTION_FROM_CMAKE_2
        template<typename T>
        using simdNonBool_t = StdArrayWith4Elems<T>;
        //template<typename T>
        //using simdBool_t = StdArrayWith4Elems<bool>;
#else
        template<typename T>
        using simdNonBool_t = SimdInterface<T>;
        //template<typename T>
        //using simdBool_t = SimdInterface<T>;
#endif
    };

    //lane count for any type T, using the selected SIMD backend
    template<typename T_Type>
    static constexpr size_t laneCount = SimdInterface<T_Type, selectedSIMDInfo::simdNonBool_t<T_Type>>::laneCount;

    //conforms to the SimdInterface class above
    template<typename T_Type>
    using SimdPack_t = SimdInterface<T_Type, selectedSIMDInfo::simdNonBool_t<T_Type>>;

    template<typename T_Type>
    class SimdLookupIndex{
        std::size_t m_idx;
    public:

        using type = T_Type;
        static constexpr auto simdWidth = laneCount<T_Type>;

        SimdLookupIndex (const std::size_t idx) : m_idx(idx){}

        //allow conversion to flat number, but not automatically
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit operator uint32_t() const
        {
            return m_idx*simdWidth;
        }
    };
} // namespace alpaka::lockstep
