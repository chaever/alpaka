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
        using SelectedSimdBackendTag = simdBackendTags::StdSimdNTimesTag<4>;
#else
        //GPU must use scalar simd packs
        using SelectedSimdBackendTag = simdBackendTags::ScalarSimdTag;
#endif
    } // namespace simdBackendTags

    //float, int, double etc count as packs of size 1
    template<typename T_Type>
    constexpr bool isTrivialPack_v = std::is_arithmetic_v<T_Type>;

    inline namespace trait{

        template<typename T_SimdBackend, typename T_Elem, typename T_SizeIndicator>
        struct PackTypeForBackend;

        template<typename T_SimdBackend, typename T_Type>
        struct IsPack;

        template<typename T_SimdBackend, typename T_Pack>
        struct PackTraits;
    }


    template<typename T_Pack>
    constexpr bool isPack_v = trait::IsPack<simdBackendTags::SelectedSimdBackendTag, T_Pack>::value;

    template<typename T_Pack>
    constexpr bool isNonTrivialPack_v = isPack_v<T_Pack> && !isTrivialPack_v<T_Pack>;

    template<typename T_Pack>
    using elemTOfPack_t = typename trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Pack>::Elem_t;

    template<typename T_Pack>
    using sizeIndicatorTOfPack_t = typename trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Pack>::SizeIndicator_t;

    //pack type that holds multiple elements if the hardware allows it.
    template<typename T_Elem, typename T_SizeIndicator>
    using Pack_t = typename trait::PackTypeForBackend<simdBackendTags::SelectedSimdBackendTag, T_Elem, T_SizeIndicator>::type;

    template<typename T_Pack>
    constexpr auto laneCount_v = trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Pack>::laneCount;

    //in this order so that T_To can be given while T_From is deduced
    template<typename T_To, typename T_From>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) convertPack(T_From&& src){
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_To>::convert(std::forward<T_From>(src));
    }

    template<typename T_Pack>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) loadPackUnaligned(elemTOfPack_t<T_Pack> const * const ptr){
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Pack>::loadUnaligned(ptr);
    }

    template<typename T_Pack>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void storePackUnaligned(T_Pack const pack, elemTOfPack_t<T_Pack> * const ptr){
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Pack>::storeUnaligned(pack, ptr);
    }

    template<typename T_Pack>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) getElem(T_Pack & pack, uint32_t const i){
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, std::decay_t<T_Pack>>::getElemAt(pack, i);
    }

    template<typename T_Lambda>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) makePackFromLambda(T_Lambda&& lambda){
        using lambda_return_t = std::decay_t<decltype(lambda(0u))>;
        static_assert(!std::is_same_v<bool, lambda_return_t>);
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, Pack_t<lambda_return_t, lambda_return_t> >::constructFromLambda(std::forward<T_Lambda>(lambda));
    }

    //for loading packs from non-contiguous memory
    template<typename T_Lambda>
    struct MemAccessorFunctor : public T_Lambda{
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr MemAccessorFunctor(T_Lambda const lambda):T_Lambda(lambda){}

        constexpr MemAccessorFunctor(MemAccessorFunctor const&) = default;
        constexpr MemAccessorFunctor(MemAccessorFunctor &)      = default;
        constexpr MemAccessorFunctor(MemAccessorFunctor &&)     = default;
    };

    inline namespace trait{

        template<typename T_Elem, typename T_SizeIndicator>
        struct PackTypeForBackend<simdBackendTags::ScalarSimdTag, T_Elem, T_SizeIndicator>{
            static_assert(isTrivialPack_v<T_Elem>);
            using type = T_Elem;
        };

        //fallback definition, any type counts as pack of size 1 if its arithmetic for any Simd backend
        template<typename T_SimdBackend, typename T_Type>
        struct IsPack{
            static_assert(!std::is_reference_v<T_Type>);
            static constexpr bool value = isTrivialPack_v<T_Type>;
        };

        //fallback definition, compiles only for trivial packs
        template<typename T_SimdBackend, typename T_Pack>
        struct PackTraits{

            static_assert(isTrivialPack_v<T_Pack>);
            static_assert(!std::is_reference_v<T_Pack>);
            static_assert(!std::is_const_v<T_Pack>);

            using SizeIndicator_t = T_Pack;
            using Elem_t = T_Pack;
            static constexpr auto laneCount = 1u;

            //conversion from T_From to T_Pack (brodacast/elemWiseCast)
            template<typename T_From>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto convert(T_From&& t){
                //when converting to trivial packs, the source must also be trivial as we would have to discard elements otherwise
                static_assert(isTrivialPack_v<std::decay_t<T_From>>);
                return static_cast<Elem_t>(std::forward<T_From>(t));
            }

            template<typename T_Lambda>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) constructFromLambda(T_Lambda&& lambda){
                return std::forward<T_Lambda>(lambda)(0u);
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) loadUnaligned(Elem_t const * const ptr){
                return *ptr;
            }

            template<typename T_Pack_>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr void storeUnaligned(T_Pack_&& value, Elem_t * const ptr){
                static_assert(std::is_same_v<T_Pack, std::decay_t<T_Pack_>>);
                *ptr = std::forward<T_Pack_>(value);
            }

            template<typename T_Source, std::enable_if_t<std::is_same_v<T_Pack, std::decay_t<T_Source>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) getElemAt(T_Source&& value, uint32_t const){
                return value;
            }
        };
    }

#if ALPAKA_USE_STD_SIMD

    namespace detail{

        template<typename T_SizeIndicator>
        using abi_native_t = std::experimental::simd_abi::native<T_SizeIndicator>;
        template<typename T_SizeIndicator>
        inline static constexpr auto sizeOfNativePack = std::experimental::simd<T_SizeIndicator, abi_native_t<T_SizeIndicator>>::size();
        template<typename T_SizeIndicator, uint32_t T_simdMult>
        using stdMultipliedSimdAbi_t = std::experimental::simd_abi::fixed_size<sizeOfNativePack<T_SizeIndicator>*T_simdMult>;
    }

    inline namespace trait{

        template<uint32_t T_simdMult, typename T_Elem, typename T_SizeIndicator>
        struct PackTypeForBackend<simdBackendTags::StdSimdNTimesTag<T_simdMult>, T_Elem, T_SizeIndicator>{
            inline static constexpr bool packIsMask = std::is_same_v<T_Elem, bool>;
            //make sure that only bool can have T_Elem != T_SizeIndicator
            static_assert(packIsMask || std::is_same_v<T_Elem, T_SizeIndicator>);
            static_assert(!std::is_same_v<bool, T_SizeIndicator>);

            //std::simd imposes a limit on how big a pack can be
            static_assert(detail::sizeOfNativePack<T_SizeIndicator> * T_simdMult <= std::experimental::simd_abi::max_fixed_size<T_SizeIndicator>);

            using type = std::conditional_t<packIsMask,
            std::experimental::simd_mask<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>,
            std::experimental::simd<T_Elem, detail::stdMultipliedSimdAbi_t<T_Elem, T_simdMult>>>;
        };

        template<uint32_t T_simdMult, typename T_SizeIndicator>
        struct IsPack<simdBackendTags::StdSimdNTimesTag<T_simdMult>, std::experimental::simd_mask<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>>{
            static constexpr bool value = true;
        };

        template<uint32_t T_simdMult, typename T_Elem>
        struct IsPack<simdBackendTags::StdSimdNTimesTag<T_simdMult>, std::experimental::simd<T_Elem, detail::stdMultipliedSimdAbi_t<T_Elem, T_simdMult>>>{
            static constexpr bool value = true;
        };



        template<uint32_t T_simdMult, typename T_Elem>
        struct PackTraits<simdBackendTags::StdSimdNTimesTag<T_simdMult>, std::experimental::simd<T_Elem, detail::stdMultipliedSimdAbi_t<T_Elem, T_simdMult>>>{

            static_assert(!std::is_reference_v<T_Elem>);

            using stdMultipliedSimdAbiPack_t = std::experimental::simd<T_Elem, detail::stdMultipliedSimdAbi_t<T_Elem, T_simdMult>>;
            using stdMultipliedSimdAbiMask_t = std::experimental::simd_mask<T_Elem, detail::stdMultipliedSimdAbi_t<T_Elem, T_simdMult>>;

            using SizeIndicator_t = T_Elem;
            using Elem_t = T_Elem;
            static constexpr auto laneCount = stdMultipliedSimdAbiPack_t::size();

            //broadcast
            template<typename T_From, std::enable_if_t<isTrivialPack_v<T_From>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& elem){
                return stdMultipliedSimdAbiPack_t(static_cast<Elem_t>(std::forward<T_From>(elem)));
            }

            //elem-wise cast
            template<typename T_From, std::enable_if_t<std::experimental::is_simd_v<std::decay_t<T_From>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& pack){
                static_assert(laneCount == std::decay_t<T_From>::size());
                return std::experimental::static_simd_cast<Elem_t, elemTOfPack_t<std::decay_t<T_From>>, detail::stdMultipliedSimdAbi_t<Elem_t, T_simdMult>>(std::forward<T_From>(pack));
            }

            //expand mask
            template<typename T_From, std::enable_if_t<std::experimental::is_simd_mask_v<std::decay_t<T_From>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& mask){
                static_assert(laneCount == std::decay_t<T_From>::size());
                stdMultipliedSimdAbiPack_t tmp(0);
                std::experimental::where(std::forward<T_From>(mask), tmp) = stdMultipliedSimdAbiPack_t(1);
                return tmp;
            }

            template<typename T_Lambda>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) constructFromLambda(T_Lambda&& lambda){
                return stdMultipliedSimdAbiPack_t(std::forward<T_Lambda>(lambda));
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) loadUnaligned(Elem_t const * const ptr){
                return stdMultipliedSimdAbiPack_t{ptr, std::experimental::element_aligned};
            }

            template<typename T_Source, std::enable_if_t<std::is_same_v<stdMultipliedSimdAbiPack_t, std::decay_t<T_Source>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr void storeUnaligned(T_Source&& value, Elem_t * const ptr){
                std::forward<T_Source>(value).copy_to(ptr, std::experimental::element_aligned);
            }

            template<typename T_Source, std::enable_if_t<std::is_same_v<stdMultipliedSimdAbiPack_t, std::decay_t<T_Source>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) getElemAt(T_Source&& value, uint32_t const i){
                return std::forward<T_Source>(value)[i];
            }
        };

        template<uint32_t T_simdMult, typename T_SizeIndicator>
        struct PackTraits<simdBackendTags::StdSimdNTimesTag<T_simdMult>, std::experimental::simd_mask<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>>{

            static_assert(!std::is_reference_v<T_SizeIndicator>);

            using stdMultipliedSimdAbiPack_t = std::experimental::simd<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>;
            using stdMultipliedSimdAbiMask_t = std::experimental::simd_mask<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>;

            using SizeIndicator_t = T_SizeIndicator;
            using Elem_t = bool;
            static constexpr auto laneCount = stdMultipliedSimdAbiPack_t::size();

            //brodacast
            template<typename T_From, std::enable_if_t<isTrivialPack_v<T_From>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& elem){
                return stdMultipliedSimdAbiMask_t{elem!=0};
            }

            //elem-wise cast
            template<typename T_From, std::enable_if_t<std::experimental::is_simd_v<std::decay_t<T_From>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& pack){
                return std::forward<T_From>(pack) != 0;
            }

            //mask->mask is trivial
            template<typename T_From, std::enable_if_t<std::experimental::is_simd_mask_v<std::decay_t<T_From>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) convert(T_From&& mask){
                static_assert(laneCount == std::decay_t<T_From>::size());
                return std::forward<T_From>(mask);
            }

            template<typename T_Lambda>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) constructFromLambda(T_Lambda&& lambda){
                static_assert(std::is_convertible_v<decltype(std::forward<T_Lambda>(lambda)(0u)), const bool>);
                return stdMultipliedSimdAbiMask_t(std::forward<T_Lambda>(lambda));
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) loadUnaligned(bool const * const ptr){
                return stdMultipliedSimdAbiMask_t{ptr, std::experimental::element_aligned};
            }

            template<typename T_Source, std::enable_if_t<std::is_same_v<stdMultipliedSimdAbiMask_t, std::decay_t<T_Source>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr void storeUnaligned(T_Source&& value, bool * const ptr){
                std::forward<T_Source>(value).copy_to(ptr, std::experimental::element_aligned);
            }

            template<typename T_Source, std::enable_if_t<std::is_same_v<stdMultipliedSimdAbiMask_t, std::decay_t<T_Source>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) getElemAt(T_Source&& value, uint32_t const i){
                return std::forward<T_Source>(value)[i];
            }
        };
    } // namespace trait

#endif

    namespace trait{

        template<typename T_Left, typename T_Right>
        constexpr bool bothArePacksAndOneIsNonTrivial_v = isPack_v<T_Left> && isPack_v<T_Right> && (isNonTrivialPack_v<T_Left>||isNonTrivialPack_v<T_Right>);

        template<typename T_Left, typename T_Right, bool T_BothPacks>
        struct PackOperatorRequirements;

        template<typename T_Left, typename T_Right>
        struct PackOperatorRequirements<T_Left, T_Right, true>{
            //cant have the same base type left & right, otherwise the operation is already defined
            constexpr static bool value = !std::is_same_v<elemTOfPack_t<T_Left>, elemTOfPack_t<T_Right>>;
        };

        template<typename T_Left, typename T_Right>
        struct PackOperatorRequirements<T_Left, T_Right, false>{
            constexpr static bool value = false;
        };
    } // namespace trait

    template<typename T_Left, typename T_Right>
    constexpr bool packOperatorRequirements_v = trait::PackOperatorRequirements<std::decay_t<T_Left>, std::decay_t<T_Right>, trait::bothArePacksAndOneIsNonTrivial_v<std::decay_t<T_Left>, std::decay_t<T_Right>>>::value;

#define XPR_OP_WRAPPER() operator

#define BINARY_READONLY_ARITHMETIC_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, sizeIndicatorTOfPack_t<std::decay_t<T_Left>>, result_elem_t>;\
        using resultPackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        using resultPackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        return convertPack<resultPackLeft_t>(std::forward<T_Left>(left)) opName convertPack<resultPackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_ARITHMETIC_OP_NO_NAMESPACE(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<alpaka::lockstep::packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<alpaka::lockstep::elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<alpaka::lockstep::elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, alpaka::lockstep::sizeIndicatorTOfPack_t<std::decay_t<T_Left>>, result_elem_t>;\
        using resultPackLeft_t = std::conditional_t<alpaka::lockstep::isTrivialPack_v<std::decay_t<T_Left>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        using resultPackRight_t = std::conditional_t<alpaka::lockstep::isTrivialPack_v<std::decay_t<T_Right>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        return alpaka::lockstep::convertPack<resultPackLeft_t>(std::forward<T_Left>(left)) opName alpaka::lockstep::convertPack<resultPackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_COMPARISON_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using compare_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() + std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, sizeIndicatorTOfPack_t<std::decay_t<T_Left>>, result_elem_t>;\
        using comparePackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, compare_elem_t, Pack_t<compare_elem_t, result_sizeInd_t>>;\
        using comparePackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, compare_elem_t, Pack_t<compare_elem_t, result_sizeInd_t>>;\
        return convertPack<comparePackLeft_t>(std::forward<T_Left>(left)) opName convertPack<comparePackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_SHIFT_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, sizeIndicatorTOfPack_t<std::decay_t<T_Left>>, result_elem_t>;\
        using resultPackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        using resultPackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, std::decay_t<T_Right>, Pack_t<result_elem_t, result_sizeInd_t>>;\
        return convertPack<resultPackLeft_t>(std::forward<T_Left>(left)) opName convertPack<resultPackRight_t>(std::forward<T_Right>(right));\
    }

#define UNARY_READONLY_OP_PREFIX(opName)\
    template<typename T_Pack, std::enable_if_t<isNonTrivialPack_v<std::decay_t<T_Pack>>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName(T_Pack&& pack)\
    {\
        return opName std::forward<T_Pack>(pack);\
    }

#define UNARY_READONLY_OP_POSTFIX(opName)\
    template<typename T_Pack, std::enable_if_t<isNonTrivialPack_v<std::decay_t<T_Pack>>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName(T_Pack&& pack)\
    {\
        return opName std::forward<T_Pack>(pack);\
    }

#define UNARY_FREE_FUNCTION(fName)\
    template<typename T_Pack, std::enable_if_t<alpaka::lockstep::isNonTrivialPack_v<std::decay_t<T_Pack>>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) fName(T_Pack&& pack){\
        return fName(std::forward<T_Pack>(pack));\
    }

    //BINARY_READONLY_ARITHMETIC_OP(+)
    //BINARY_READONLY_ARITHMETIC_OP(-)
    BINARY_READONLY_ARITHMETIC_OP_NO_NAMESPACE(+)
    BINARY_READONLY_ARITHMETIC_OP_NO_NAMESPACE(-)
    BINARY_READONLY_ARITHMETIC_OP(*)
    BINARY_READONLY_ARITHMETIC_OP(/)
    BINARY_READONLY_ARITHMETIC_OP(&)
    BINARY_READONLY_ARITHMETIC_OP(&&)
    BINARY_READONLY_ARITHMETIC_OP(|)
    BINARY_READONLY_ARITHMETIC_OP(||)

    BINARY_READONLY_COMPARISON_OP(>)
    BINARY_READONLY_COMPARISON_OP(<)

    //UNARY_READONLY_OP_PREFIX(!)
    //UNARY_READONLY_OP_PREFIX(~)

    //UNARY_READONLY_OP_POSTFIX(++)
    //UNARY_READONLY_OP_POSTFIX(--)

    //BINARY_READONLY_SHIFT_OP(>>)
    //BINARY_READONLY_SHIFT_OP(<<)

} // namespace alpaka::lockstep

    //UNARY_FREE_FUNCTION(std::abs)

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
    };

    namespace trait{

        template<typename T_Idx>
        struct IsSimdLookupIdx{
            static constexpr bool value = false;
        };

        template<typename T_Foreach>
        struct IsSimdLookupIdx<SimdLookupIndex<T_Foreach>>{
            static constexpr bool value = true;
        };

        template<typename T_Idx>
        struct IsScalarLookupIdx{
            static constexpr bool value = false;
        };

        template<typename T_Foreach, uint32_t T_offset>
        struct IsScalarLookupIdx<ScalarLookupIndex<T_Foreach, T_offset>>{
            static constexpr bool value = true;
        };
    }

    template<typename T_Idx>
    constexpr auto isSimdLookupIdx_v = trait::IsSimdLookupIdx<T_Idx>::value;

    template<typename T_Idx>
    constexpr auto isScalarLookupIdx_v = trait::IsScalarLookupIdx<T_Idx>::value;

#undef ALPAKA_USE_STD_SIMD
} // namespace alpaka::lockstep
