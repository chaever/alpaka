
#include <experimental/simd>

//the following operators/functions are features missing from std::simd that are needed.
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
