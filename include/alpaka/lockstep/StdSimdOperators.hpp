
#include <experimental/simd>

//the following operators/functions are features missing from std::simd that are needed.
//if you get a compiler error that mentions that any of these functions are being re-declared/re-defined, delete the function as it is superfluous at that point.

namespace std::experimental
{
    //specific for std::simd, allows addition of Pack<T> and Pack<T>::mask which is not normally possible
    //should still be findable through ADL

    //for some reason, uint + pack<uint>  or  char * pack<char> is defined, but   bool && pack<bool>(aka mask) is missing
    template<typename T_Left, std::enable_if_t<std::experimental::is_simd_mask_v<std::decay_t<T_Left>>, int> = 0>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator&&(T_Left&& left, bool const right)
    {
        return std::decay_t<T_Left>(right) && std::forward<T_Left>(left);
    }

    template<typename T_Right, std::enable_if_t<std::experimental::is_simd_mask_v<std::decay_t<T_Right>>, int> = 0>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator&&(bool const left, T_Right&& right)
    {
        //re-use other operator definition
        return std::forward<T_Right>(right) && left;
    }
}

//std::abs for floating-point-based simd-packs (currently not supported by default)
template <typename T_Pack, std::enable_if_t<std::experimental::is_simd_v<std::decay_t<T_Pack>>, int> = 0>
ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) std::abs(T_Pack&& floatPack)
{
    using Pack = std::decay_t<T_Pack>;
    //if we are dealing with unsigned values we are extremely likely to overflow which violates the assumption that abs will always be positive
    static_assert(std::is_signed_v<typename Pack::value_type>);
    Pack tmp{std::forward<T_Pack>(floatPack)};
    std::experimental::where(std::forward<T_Pack>(floatPack) < 0, tmp) *= Pack(-1);
    return tmp;
}
