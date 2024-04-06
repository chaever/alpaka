///TODO Copyright, License and other stuff here

#pragma once

#include <experimental/simd>
#include <algorithm>

//the following 2 operators/functions are features missing from std::simd that are needed.
//if you get a compiler error that mentions that any of these functions are being re-declared/re-defined, delete the function as it is superfluous at that point.

namespace std::experimental
{
    //specific for std::simd, allows addition of Pack<T> and Pack<T>::mask which is not normally possible
    //should still be findable through ADL
    template<typename T_Elem, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::enable_if_t<std::is_arithmetic_v<T_Elem>, std::experimental::simd<T_Elem, T_Abi>> operator+(std::experimental::simd<T_Elem, T_Abi> const& left, std::experimental::simd_mask<T_Elem, T_Abi> const& right)
    {
        using Pack = std::experimental::simd<T_Elem, T_Abi>;
        ///TODO once std::experimental::where supports it, make this constexpr
        /*constexpr*/ Pack tmp(left);
        std::experimental::where(right, tmp) += Pack(1);
        return tmp;
    }

    template<typename T_Elem, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator+(std::experimental::simd_mask<T_Elem, T_Abi> const& left, std::experimental::simd<T_Elem, T_Abi> const& right)
    {
        //re-use other operator definition
        return right+left;
    }
}

//std::abs for floating-point-based simd-packs (currently not supported by default)
template <typename T_Elem, typename T_Abi>
std::enable_if_t<std::is_floating_point_v<T_Elem> && std::is_signed_v<T_Elem>, std::experimental::simd<T_Elem, T_Abi>>
std::abs(const std::experimental::simd<T_Elem, T_Abi>& floatPack)
{
    using Pack = std::experimental::simd<T_Elem, T_Abi>;
    Pack tmp{floatPack};
    std::experimental::where(floatPack < 0, tmp) = Pack(-1) * floatPack;
    return tmp;
}

namespace alpaka::lockstep
{
    //provides Information about the pack framework the user selected via CMake
    namespace simdBackendTags{

        class ScalarSimdTag{};
        class ArrayOf4Tag{};
        class StdSimdTag{};
        template<uint32_t T_simdMult>
        class StdSimdNTimesTag{};

#if   0 || defined COMPILE_OPTION_FROM_CMAKE_1
        using SelectedSimdBackendTag = simdBackendTags::StdSimdTag;
#elif 0 || defined COMPILE_OPTION_FROM_CMAKE_2
        using SelectedSimdBackendTag = simdBackendTags::ArrayOf4Tag;
#elif 1 || defined COMPILE_OPTION_FROM_CMAKE_3
        using SelectedSimdBackendTag = simdBackendTags::StdSimdNTimesTag<2>;
#else
        using SelectedSimdBackendTag = simdBackendTags::ScalarSimdTag;
#endif
    } // namespace simdBackendTags

    template<typename T_Elem, typename T_SizeIndicator, typename T_Simd>
    struct SimdInterface;

    //conforms to the SimdInterface class above
    template<typename T_Type, typename T_SizeIndicator>
    using SimdInterface_t = SimdInterface<T_Type, T_SizeIndicator, simdBackendTags::SelectedSimdBackendTag>;

    //lane count for any type T, using the selected SIMD backend
    template<typename T_SizeIndicator>
    static constexpr size_t laneCount_v = SimdInterface_t<T_SizeIndicator, T_SizeIndicator>::laneCount;

    template<typename T_Type, typename T_SizeIndicator>
    using Pack_t = typename SimdInterface_t<T_Type, T_SizeIndicator>::Pack_t;

    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem, typename T_SizeIndicator>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::ScalarSimdTag>{
        using Elem_t = T_Elem;
        using Pack_t = T_Elem;//special case laneCount=1 : elements are packs

        inline static constexpr std::size_t laneCount = 1u;

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(const T_Elem* const mem) -> Pack_t
        {
            return *mem;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, T_Elem* const mem)
        {
            *mem = t;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto broadcast(T_Elem const & elem) -> Pack_t
        {
            return elem;
        }

        template<typename Source_t>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(Source_t const& pack) -> Pack_t
        {
            return static_cast<T_Elem>(pack);
        }
    };

    //Specialization using std::experimental::simd
    //in case of T_Elem = bool, the size of the packs is decided by T_SizeIndicator
    template<typename T_Elem, typename T_SizeIndicator>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::StdSimdTag>{
        inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;

        //make sure that only bool can have T_Elem != T_SizeIndicator
        static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
        static_assert(!std::is_same_v<bool, T_SizeIndicator>);

        using Elem_t = T_Elem;
        using abi_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        using Pack_t = std::conditional_t<packIsMask,
        std::experimental::simd_mask<T_SizeIndicator, abi_t>,
        std::experimental::simd<T_Elem, abi_t>>;

        inline static constexpr std::size_t laneCount = Pack_t::size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(T_Elem const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, T_Elem * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto broadcast(T_Elem const& elem) -> Pack_t
        {
            return Pack_t(elem);
        }

        //got pack, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            return std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack);
        }

        //got mask, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            static_assert(std::is_arithmetic_v<T_Source_Elem>);
            Pack_t tmp(0);
            std::experimental::where(mask, tmp) = Pack_t(1);
            return tmp;
        }

        //got pack, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            //arithmetic types casted to bool are true if != 0
            return pack != 0;
        }

        //got mask, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            return mask;
        }
    };

    //std::experimental::simd, but N at a time
    template<typename T_Elem, typename T_SizeIndicator, uint32_t T_simdMult>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
        inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;

        //make sure that only bool can have T_Elem != T_SizeIndicator
        static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
        static_assert(!std::is_same_v<bool, T_SizeIndicator>);
        static_assert(T_simdMult>1u, "Tried to use StdSimdNTimesTag<T_simdMult> with T_simdMult=1. Use StdSimdTag in this case.");

        using Elem_t = T_Elem;
        using abi_native_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        inline static constexpr auto sizeOfNativePack = std::experimental::simd<T_SizeIndicator, abi_native_t>::size();
        using abi_multiplied_t = std::experimental::simd_abi::fixed_size<sizeOfNativePack*T_simdMult>;
        using Pack_t = std::conditional_t<packIsMask,
        std::experimental::simd_mask<T_SizeIndicator, abi_multiplied_t>,
        std::experimental::simd<T_Elem, abi_multiplied_t>>;

        inline static constexpr std::size_t laneCount = Pack_t::size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto loadUnaligned(T_Elem const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC void storeUnaligned(const Pack_t t, T_Elem * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto broadcast(T_Elem const & elem) -> Pack_t
        {
            return Pack_t(elem);
        }

        //got pack, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            return std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack);
        }

        //got mask, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            static_assert(std::is_arithmetic_v<T_Source_Elem>);
            Pack_t tmp(0);
            std::experimental::where(mask, tmp) = Pack_t(1);
            return tmp;
        }

        //got pack, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            //arithmetic types casted to bool are true if != 0
            return pack != 0;
        }

        //got mask, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            return mask;
        }
    };

    template<typename T_Elem>
    using StdArrayWith4Elems = std::array<T_Elem, alpaka::core::vectorization::GetVectorizationSizeElems<T_Elem>::value>;

    //Specialization using 4 values in an array
    template<typename T_Elem, typename T_SizeIndicator>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::ArrayOf4Tag>{
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

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto broadcast(T_Elem const & elem) -> Pack_t
        {
            Pack_t tmp;
            for(auto i=0u; i<laneCount; ++i)
                tmp[i] = elem;
            return tmp;
        }

        template<typename Source_t>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC auto elementWiseCast(Source_t const& pack) -> Pack_t
        {
            Pack_t tmp;
            for(auto i=0u; i<laneCount; ++i)
                tmp[i] = static_cast<T_Elem>(pack[i]);
            return tmp;
        }
    };

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

    //for accessing the value of scalar Nodes (that only have one value)
    class SingleElemIndex{};
} // namespace alpaka::lockstep
