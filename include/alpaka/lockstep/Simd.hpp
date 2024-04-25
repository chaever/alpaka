///TODO Copyright, License and other stuff here

#pragma once

//on the CPU, we should have std::simd.
#if defined ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED || defined ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED || defined ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED

#define ALPAKA_USE_STD_SIMD 1

#include <experimental/simd>

//the following 2 operators/functions are features missing from std::simd that are needed.
//if you get a compiler error that mentions that any of these functions are being re-declared/re-defined, delete the function as it is superfluous at that point.

namespace std::experimental
{
    //specific for std::simd, allows addition of Pack<T> and Pack<T>::mask which is not normally possible
    //should still be findable through ADL
    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem_Left, T_Abi> operator+(std::experimental::simd<T_Elem_Left, T_Abi> const& left, std::experimental::simd_mask<T_Elem_Right, T_Abi> const& right)
    {
        using Pack = std::experimental::simd<T_Elem_Left, T_Abi>;
        Pack tmp(left);
        std::experimental::where(right, tmp) += Pack(1);
        return tmp;
    }

    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem_Left, T_Abi> operator-(std::experimental::simd<T_Elem_Left, T_Abi> const& left, std::experimental::simd_mask<T_Elem_Right, T_Abi> const& right)
    {
        using Pack = std::experimental::simd<T_Elem_Left, T_Abi>;
        Pack tmp(left);
        std::experimental::where(right, tmp) -= Pack(1);
        return tmp;
    }

    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem_Left, T_Abi> operator*(std::experimental::simd<T_Elem_Left, T_Abi> const& left, std::experimental::simd_mask<T_Elem_Right, T_Abi> const& right)
    {
        using Pack = std::experimental::simd<T_Elem_Left, T_Abi>;
        Pack tmp(left);
        std::experimental::where(!right, tmp) = Pack(0);
        return tmp;
    }

    template<typename T_Elem_Left, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd_mask<T_Elem_Left, T_Abi> operator&&(std::experimental::simd_mask<T_Elem_Left, T_Abi> const& left, bool const& right)
    {
        std::experimental::simd_mask<T_Elem_Left, T_Abi> tmp(right);
        return tmp && left;
    }

    template<typename T_Elem, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem, T_Abi> operator+(std::experimental::simd<T_Elem, T_Abi> const& left, T_Elem const& right)
    {
        return left + std::experimental::simd<T_Elem, T_Abi>(right);
    }

    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator+(std::experimental::simd_mask<T_Elem_Left, T_Abi> const& left, std::experimental::simd<T_Elem_Right, T_Abi> const& right)
    {
        //re-use other operator definition
        return right+left;
    }

    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator-(std::experimental::simd_mask<T_Elem_Left, T_Abi> const& left, std::experimental::simd<T_Elem_Right, T_Abi> const& right)
    {
        //re-use other operator definition
        return right-left;
    }

    template<typename T_Elem_Left, typename T_Elem_Right, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator*(std::experimental::simd_mask<T_Elem_Left, T_Abi> const& left, std::experimental::simd<T_Elem_Right, T_Abi> const& right)
    {
        //re-use other operator definition
        return right*left;
    }

    template<typename T_Elem_Left, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator&&(bool const& left, std::experimental::simd_mask<T_Elem_Left, T_Abi> const& right)
    {
        //re-use other operator definition
        return right&&left;
    }

    template<typename T_Elem, typename T_Abi>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem, T_Abi> operator+(T_Elem const& left, std::experimental::simd<T_Elem, T_Abi> const& right)
    {
        //re-use other operator definition
        return right+left;
    }
}

//std::abs for floating-point-based simd-packs (currently not supported by default)
template <typename T_Elem, typename T_Abi>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr std::experimental::simd<T_Elem, T_Abi> std::abs(const std::experimental::simd<T_Elem, T_Abi>& floatPack)
{
    //if we are dealing with unsigned values we are extremely likely to overflow which violates the assumption that abs will always be positive
    static_assert(std::is_signed_v<T_Elem>);
    using Pack = std::experimental::simd<T_Elem, T_Abi>;
    Pack tmp{floatPack};
    std::experimental::where(floatPack < 0, tmp) = Pack(-1) * floatPack;
    return tmp;
}

#endif

namespace alpaka::lockstep
{
    //provides Information about the pack framework the user selected via CMake
    namespace simdBackendTags{

        class ScalarSimdTag{};
        class StdSimdTag{};
        template<uint32_t T_simdMult>
        class StdSimdNTimesTag{};

#if   0 && ALPAKA_USE_STD_SIMD
        using SelectedSimdBackendTag = simdBackendTags::StdSimdTag;
#elif 1 && ALPAKA_USE_STD_SIMD
        using SelectedSimdBackendTag = simdBackendTags::StdSimdNTimesTag<1>;
#else
        //GPU must use scalar simd packs
        using SelectedSimdBackendTag = simdBackendTags::ScalarSimdTag;
#endif
    } // namespace simdBackendTags

    template<typename T_Elem, typename T_SizeIndicator, typename T_Simd>
    struct SimdInterface;

    template<typename T_Pack, typename T_Simd>
    struct GetSizeIndicator;

    //conforms to the SimdInterface class above
    template<typename T_Type, typename T_SizeIndicator>
    using SimdInterface_t = SimdInterface<T_Type, T_SizeIndicator, simdBackendTags::SelectedSimdBackendTag>;

    //lane count for any type T, using the selected SIMD backend
    template<typename T_SizeIndicator>
    static constexpr size_t laneCount_v = SimdInterface_t<T_SizeIndicator, T_SizeIndicator>::laneCount;

    template<typename T_Type, typename T_SizeIndicator>
    using Pack_t = typename SimdInterface_t<T_Type, T_SizeIndicator>::Pack_t;

    template<typename T>
    using GetSizeIndicator_t = typename GetSizeIndicator<std::decay_t<T>, simdBackendTags::SelectedSimdBackendTag>::type;

    //General-case SIMD interface, corresponds to a simdWidth of 1
    template<typename T_Elem, typename T_SizeIndicator>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::ScalarSimdTag>{
        using Elem_t = T_Elem;
        using Pack_t = T_Elem;//special case laneCount=1 : elements are packs

        inline static constexpr std::size_t laneCount = 1u;

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto loadUnaligned(const T_Elem* const mem) -> Pack_t
        {
            return *mem;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr void storeUnaligned(const Pack_t t, T_Elem* const mem)
        {
            *mem = t;
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto broadcast(T_Elem const & elem) -> Pack_t
        {
            return elem;
        }

        template<typename Source_t>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(Source_t const& pack) -> Pack_t
        {
            return static_cast<T_Elem>(pack);
        }
    };

#if ALPAKA_USE_STD_SIMD

    template<typename T_SizeIndicator>
    struct GetSizeIndicator<T_SizeIndicator, simdBackendTags::ScalarSimdTag>{
        using type = T_SizeIndicator;
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

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto loadUnaligned(T_Elem const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr void storeUnaligned(const Pack_t t, T_Elem * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto broadcast(T_Elem const& elem) -> Pack_t
        {
            return Pack_t(elem);
        }

        //got pack, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) // -> Pack_t
        {
            constexpr bool isTrivial=std::is_same_v<T_Source_Elem, T_Elem>;
            static_assert(isTrivial != (std::is_same_v<T_Source_Elem, uint32_t> && std::is_same_v<T_Elem, float>));


            {
                static_assert(isTrivial != (std::is_same_v<T_Source_Elem, uint32_t> && std::is_same_v<T_Elem, float>));

                using to = float;
                using from = uint32_t;


                using pUint = std::experimental::simd<from, std::experimental::simd_abi::native<from> >;
                using pFloat = std::experimental::simd<to, std::experimental::simd_abi::native<to> >;

                pUint someUints{6u};
                pFloat someFloats{5.0f};

                auto tmp = std::experimental::static_simd_cast<to, from, std::experimental::simd_abi::native<to>>(someUints);

                pFloat tmp2 = std::experimental::to_native(tmp);


                static_assert(isTrivial || (std::is_same_v<T_Source_Abi, abi_t> && std::is_same_v<abi_t, std::experimental::simd_abi::native<T_Elem>>));
                static_assert(isTrivial || (std::is_same_v<pUint, std::decay_t<decltype(pack)>>));
            }







            //auto tmp = std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack);
            auto tmp = std::experimental::static_simd_cast<T_Elem, T_Source_Elem, std::experimental::simd_abi::native<T_Elem>>(pack);



            static_assert(std::decay_t<decltype(std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack))>::size() == std::decay_t<decltype(pack)>::size());
            static_assert(std::is_same_v<T_Source_Abi, abi_t>);
            static_assert(isTrivial || (std::is_same_v<typename std::decay_t<decltype(tmp )>::abi_type, std::experimental::simd_abi::_Fixed<4> > ));
            static_assert(isTrivial || (std::is_same_v<T_Source_Abi, std::experimental::simd_abi::_VecBuiltin<16> > ));
            static_assert(isTrivial || (std::is_same_v<T_Source_Abi, typename std::decay_t<decltype(pack)>::abi_type>));

            static_assert(isTrivial || (std::is_same_v<std::decay_t<decltype(pack)>, std::experimental::simd<uint32_t, std::experimental::simd_abi::_VecBuiltin<16> > >));
            static_assert(isTrivial || (std::is_same_v<decltype(tmp ), std::experimental::simd<float, std::experimental::simd_abi::_Fixed<4> > >));
            //static_assert(isTrivial || (std::is_same_v<Pack_t, alpaka::lockstep::Pack_t<float, float>>));

            //alpaka::lockstep::Pack_t<T_Elem, T_SizeIndicator> tmp2 = std::experimental::to_native(tmp);



            //return /*std::experimental::to_native(*/std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack)/*)*/;
            //Pack_t tmp2 = std::experimental::to_native(tmp);
            return tmp;
        }

        //got mask, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            static_assert(std::is_arithmetic_v<T_Source_Elem>);
            Pack_t tmp(0);
            std::experimental::where(mask, tmp) = Pack_t(1);
            return tmp;
        }

        //got pack, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            //arithmetic types casted to bool are true if != 0
            return pack != 0;
        }

        //got mask, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            return mask;
        }
    };

    template<typename T_SizeIndicator, typename T_Abi>
    struct GetSizeIndicator<std::experimental::simd_mask<T_SizeIndicator, T_Abi>, simdBackendTags::StdSimdTag>{
        using type = T_SizeIndicator;
    };

    template<typename T_SizeIndicator, typename T_Abi>
    struct GetSizeIndicator<std::experimental::simd<T_SizeIndicator, T_Abi>, simdBackendTags::StdSimdTag>{
        using type = T_SizeIndicator;
    };

    //std::experimental::simd, but N at a time
    template<typename T_Elem, typename T_SizeIndicator, uint32_t T_simdMult>
    struct SimdInterface<T_Elem, T_SizeIndicator, simdBackendTags::StdSimdNTimesTag<T_simdMult>>{
        inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;

        //make sure that only bool can have T_Elem != T_SizeIndicator
        static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
        static_assert(!std::is_same_v<bool, T_SizeIndicator>);

        using Elem_t = T_Elem;
        using abi_native_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        inline static constexpr auto sizeOfNativePack = std::experimental::simd<T_SizeIndicator, abi_native_t>::size();
        using abi_multiplied_t = std::experimental::simd_abi::fixed_size<sizeOfNativePack*T_simdMult>;
        using Pack_t = std::conditional_t<packIsMask,
        std::experimental::simd_mask<T_SizeIndicator, abi_multiplied_t>,
        std::experimental::simd<T_Elem, abi_multiplied_t>>;

        inline static constexpr std::size_t laneCount = Pack_t::size();

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto loadUnaligned(T_Elem const * const mem) -> Pack_t
        {
            return Pack_t(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr void storeUnaligned(const Pack_t t, T_Elem * const mem)
        {
            t.copy_to(mem, std::experimental::element_aligned);
        }

        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto broadcast(T_Elem const & elem) -> Pack_t
        {
            return Pack_t(elem);
        }

        //got pack, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            return std::experimental::static_simd_cast<T_Elem, T_Source_Elem, T_Source_Abi>(pack);
        }

        //got mask, own elements are non-bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount && !packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            static_assert(std::is_arithmetic_v<T_Source_Elem>);
            Pack_t tmp(0);
            std::experimental::where(mask, tmp) = Pack_t(1);
            return tmp;
        }

        //got pack, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd<T_Source_Elem, T_Source_Abi> const& pack) -> Pack_t
        {
            //arithmetic types casted to bool are true if != 0
            return pack != 0;
        }

        //got mask, own elements are bool
        template<typename T_Source_Elem, typename T_Source_Abi, std::enable_if_t<laneCount_v<T_Source_Elem> == laneCount &&  packIsMask, int> = 0>
        static ALPAKA_FN_INLINE ALPAKA_FN_HOST_ACC constexpr auto elementWiseCast(std::experimental::simd_mask<T_Source_Elem, T_Source_Abi> const& mask) -> Pack_t
        {
            return mask;
        }
    };

    template<typename T_SizeIndicator, typename T_Abi, uint32_t T_mult>
    struct GetSizeIndicator<std::experimental::simd_mask<T_SizeIndicator, T_Abi>, simdBackendTags::StdSimdNTimesTag<T_mult>>{
        using type = T_SizeIndicator;
    };

    template<typename T_SizeIndicator, typename T_Abi, uint32_t T_mult>
    struct GetSizeIndicator<std::experimental::simd<T_SizeIndicator, T_Abi>, simdBackendTags::StdSimdNTimesTag<T_mult>>{
        using type = T_SizeIndicator;
    };

#endif

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
