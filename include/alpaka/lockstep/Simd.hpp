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

    template<typename T_Dest, typename T_BoolOrMask>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) conditionalAssignable(T_Dest& dest, T_BoolOrMask const& select){
        static_assert(!std::is_const_v<T_Dest>);
        static_assert(isPack_v<T_Dest>);
        static_assert(std::is_same_v<bool, elemTOfPack_t<T_BoolOrMask>>);
        return trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, T_Dest>::elemWiseConditionalAssignable(dest, select);
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

        template<typename T_Dest>
        struct ConditionallyAssignableReference{
            T_Dest & dest;
            const bool dontIgnoreAssignmnet;

            template<typename T>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void operator= (T&& value) {
                if(dontIgnoreAssignmnet)
                {
                    dest = std::forward<T>(value);
                }
            }

#define ASSIGN_OP(op)\
            template<typename T>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void operator op (T&& value) {\
                if(dontIgnoreAssignmnet)\
                {\
                    dest op std::forward<T>(value);\
                }\
            }

            ASSIGN_OP(+=)
            ASSIGN_OP(-=)
            ASSIGN_OP(*=)
            ASSIGN_OP(/=)
            ASSIGN_OP(%=)

#undef ASSIGN_OP
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
                return std::forward<T_Source>(value);
            }

            ///TODO tmp modification returns by value, forced
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto elemWiseConditionalAssignable(Elem_t & dest, bool const select){
                return ConditionallyAssignableReference(dest, select);
            }
        };

        template<typename T_Pack, bool T_IsPack>
        struct IsMask;

        template<typename T_Pack>
        struct IsMask<T_Pack, false>{
            constexpr static bool value = false;
        };

        template<typename T_Pack>
        struct IsMask<T_Pack, true>{
            constexpr static bool value = std::is_same_v<bool, elemTOfPack_t<std::decay_t<T_Pack> > >;
        };
    } // namespace trait

    template<typename T_Pack>
    constexpr bool isMask_v = trait::IsMask<std::decay_t<T_Pack>, isPack_v<std::decay_t<T_Pack>> >::value;

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

            ///TODO tmp modification returns by value, forced
            template<typename T_Mask>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto elemWiseConditionalAssignable(stdMultipliedSimdAbiPack_t & dest, T_Mask const& select){
#if 1
                static_assert(isMask_v<T_Mask>);
                auto forReturn = std::experimental::where(select, dest);
                std::cout << "trait::elemWiseConditionalAssignable  :address of pack is " << reinterpret_cast<uint64_t>(&dest) << std::endl;
                std::cout << "trait::elemWiseConditionalAssignable  :address of mask is " << reinterpret_cast<uint64_t>(&select) << std::endl;
                std::cout << "trait::elemWiseConditionalAssignable  :where object bytewise      ";
                for(auto i=0u;i<sizeof(std::decay_t<decltype(forReturn)>);++i){
                    std::cout << std::setw(4) << static_cast<uint32_t>(reinterpret_cast<char*>(&forReturn)[i]) << " ";
                }
                std::cout<< std::endl;
#endif
                return std::experimental::where(select, dest);
            }
        };

        template<uint32_t T_simdMult, typename T_SizeIndicator>
        struct PackTraits<simdBackendTags::StdSimdNTimesTag<T_simdMult>, std::experimental::simd_mask<T_SizeIndicator, detail::stdMultipliedSimdAbi_t<T_SizeIndicator, T_simdMult>>>{

            static_assert(!std::is_reference_v<T_SizeIndicator>);
            static_assert(!std::is_same_v<T_SizeIndicator, bool>);

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
            constexpr static bool value = !std::is_same_v<elemTOfPack_t<T_Left>, elemTOfPack_t<T_Right>> || std::is_same_v<elemTOfPack_t<T_Left>, bool>;
        };

        template<typename T_Left, typename T_Right>
        struct PackOperatorRequirements<T_Left, T_Right, false>{
            constexpr static bool value = false;
        };

        template<bool rev>
        struct ConditionallyReverse;

        template<>
        struct ConditionallyReverse<false>{
            template<typename T_1, typename T_2>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getFirst(T_1&& one, T_2&&){
                return std::forward<T_1>(one);
            }
            template<typename T_1, typename T_2>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getSecond(T_1&&, T_2&& two){
                return std::forward<T_2>(two);
            }
        };

        template<>
        struct ConditionallyReverse<true>{
            template<typename T_1, typename T_2>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getFirst(T_1&&, T_2&& two){
                return std::forward<T_2>(two);
            }
            template<typename T_1, typename T_2>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr auto getSecond(T_1&& one, T_2&&){
                return std::forward<T_1>(one);
            }
        };

    } // namespace trait

    template<typename T_Left, typename T_Right>
    constexpr bool packOperatorRequirements_v = trait::PackOperatorRequirements<std::decay_t<T_Left>, std::decay_t<T_Right>, trait::bothArePacksAndOneIsNonTrivial_v<std::decay_t<T_Left>, std::decay_t<T_Right>>>::value;

    template<typename T_Left, typename T_Right, typename T_Result>
    using packOperatorSizeInd_t = std::conditional_t<!std::is_same_v<bool, T_Result>, T_Result, std::conditional_t<!std::is_same_v<bool, sizeIndicatorTOfPack_t<T_Left> > , sizeIndicatorTOfPack_t<T_Left>, sizeIndicatorTOfPack_t<T_Right> > >;

    template<typename T_Left, typename T_Right>
    constexpr bool hasExactlyOneMask_v = std::is_same_v<bool, elemTOfPack_t<std::decay_t<T_Left> > > ^ std::is_same_v<bool, elemTOfPack_t<std::decay_t<T_Right> > >;

#define XPR_OP_WRAPPER() operator

#define BINARY_READONLY_ARITHMETIC_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = packOperatorSizeInd_t<std::decay_t<T_Left>, std::decay_t<T_Right>, result_elem_t>;\
        static_assert(!std::is_same_v<bool, result_sizeInd_t>);\
        using resultPackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        using resultPackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        return convertPack<resultPackLeft_t>(std::forward<T_Left>(left)) opName convertPack<resultPackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_ARITHMETIC_OP_MASK_EXCLUDED(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right> && (isMask_v<T_Left> == isMask_v<T_Right>), int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = packOperatorSizeInd_t<std::decay_t<T_Left>, std::decay_t<T_Right>, result_elem_t>;\
        static_assert(!std::is_same_v<bool, result_sizeInd_t>);\
        using resultPackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        using resultPackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, result_elem_t, Pack_t<result_elem_t, result_sizeInd_t>>;\
        return convertPack<resultPackLeft_t>(std::forward<T_Left>(left)) opName convertPack<resultPackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_COMPARISON_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using compare_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() + std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = packOperatorSizeInd_t<std::decay_t<T_Left>, std::decay_t<T_Right>, result_elem_t>;\
        using comparePackLeft_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Left>>, compare_elem_t, Pack_t<compare_elem_t, result_sizeInd_t>>;\
        using comparePackRight_t = std::conditional_t<isTrivialPack_v<std::decay_t<T_Right>>, compare_elem_t, Pack_t<compare_elem_t, result_sizeInd_t>>;\
        return convertPack<comparePackLeft_t>(std::forward<T_Left>(left)) opName convertPack<comparePackRight_t>(std::forward<T_Right>(right));\
    }

#define BINARY_READONLY_SHIFT_OP(opName)\
    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()opName (T_Left&& left, T_Right&& right){\
        using result_elem_t = decltype(std::declval<elemTOfPack_t<std::decay_t<T_Left>>>() opName std::declval<elemTOfPack_t<std::decay_t<T_Right>>>());\
        using result_sizeInd_t = packOperatorSizeInd_t<std::decay_t<T_Left>, std::decay_t<T_Right>, result_elem_t>;\
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

    /*
    operator+(mask, pack) -> return conditionalAssign(pack, [](auto dest){dest += (1 casted to elem_t of pack);}, mask);
    operator-(mask, pack) -> return conditionalAssign(pack, [](auto dest){dest -= (1 casted to elem_t of pack);}, mask);
    operator*(mask, pack) -> return conditionalAssign(pack, [](auto dest){dest = (0 casted to elem_t of pack);}, mask);
    operator&(mask, pack) -> use current overload that expands the mask and then does the packwise &

    General Idea: have conditionalAssign return sth that can be assigned to (either ref in scalar case, or where_expr in case of std::simd.)-> loose out on bool, should be OK
    */

    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right> && (isMask_v<T_Left> ^ isMask_v<T_Right>), int> = 0>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator+ (T_Left&& left, T_Right&& right){
        using leftElem_t = elemTOfPack_t<std::decay_t<T_Left>>;
        using rightElem_t = elemTOfPack_t<std::decay_t<T_Right>>;
        constexpr bool leftIsMask = std::is_same_v<bool, std::decay_t<leftElem_t>>;
        using pack_elem_t = std::conditional_t<leftIsMask, rightElem_t, leftElem_t>;
        static_assert(!std::is_same_v<bool, std::decay_t<pack_elem_t>>);
        /*swap operands if needed*/
        auto pack = trait::ConditionallyReverse<leftIsMask>::getFirst(std::forward<T_Left>(left), std::forward<T_Right>(right));
        conditionalAssignable(pack, trait::ConditionallyReverse<leftIsMask>::getSecond(std::forward<T_Left>(left), std::forward<T_Right>(right))) += static_cast<pack_elem_t>(1);

        //std::cout << "operator+ (" << (leftIsMask?"mask, pack":"pack, mask") << ")" << std::endl;

        return pack;
    }

    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right> && (!isMask_v<T_Left> && isMask_v<T_Right>), int> = 0>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator- (T_Left&& left, T_Right&& right){
        using leftElem_t = elemTOfPack_t<std::decay_t<T_Left>>;
        auto pack = std::forward<T_Left>(left);
        auto maskCopy = std::forward<T_Right>(right);

#if 0
        conditionalAssignable(pack, maskCopy) -= static_cast<leftElem_t>(1);
#else
        auto packForWhere = pack;
        auto packForTrait = pack;


        std::cout << "operator-  :address of pack is " << reinterpret_cast<uint64_t>(&pack) << std::endl;
        std::cout << "operator-  :address of mask is " << reinterpret_cast<uint64_t>(&maskCopy) << std::endl;
        decltype(auto) where2 = conditionalAssignable(pack, maskCopy);
        std::cout << "operator-                             :where from trait bytewise  ";
        for(auto i=0u;i<sizeof(std::decay_t<decltype(where2)>);++i){
            std::cout << std::setw(4) << static_cast<uint32_t>(reinterpret_cast<char*>(&where2)[i]) << " ";
        }
        decltype(auto) where1 = std::experimental::where(maskCopy, pack);
        std::cout << "\noperator-                             :direct std::where bytewise ";
        for(auto i=0u;i<sizeof(std::decay_t<decltype(where1)>);++i){
            std::cout << std::setw(4) << static_cast<uint32_t>(reinterpret_cast<char*>(&where1)[i]) << " ";
        }
        std::cout << "\n<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;

        static_assert(std::is_same_v<std::decay_t<decltype(where1)>, std::decay_t<decltype(where2)>>);

        conditionalAssignable(pack, maskCopy) -= static_cast<leftElem_t>(1);

        std::experimental::where(maskCopy, packForWhere) -= static_cast<leftElem_t>(1);

        trait::PackTraits<simdBackendTags::SelectedSimdBackendTag, std::decay_t<decltype(packForTrait)>>::elemWiseConditionalAssignable(packForTrait, maskCopy) -= static_cast<leftElem_t>(1);

        const bool match = std::experimental::all_of(packForWhere==pack);
        if(!match){
            std::cout << "mismatch in operator-\ncorrect:    [";
            for(auto i=0u;i<laneCount_v<std::decay_t<T_Left>>;++i){
                std::cout<<packForWhere[i]<<" ";
            }
            std::cout << "]\ncalculated: [";
            for(auto i=0u;i<laneCount_v<std::decay_t<T_Left>>;++i){
                std::cout<<pack[i]<<" ";
            }
            std::cout << "]"<<std::endl;

            throw std::invalid_argument( "operator- was faulty" );
        }
#endif
        return pack;
    }

    template<typename T_Left, typename T_Right, std::enable_if_t<packOperatorRequirements_v<T_Left, T_Right> && (isMask_v<T_Left> ^ isMask_v<T_Right>), int> = 0>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator* (T_Left&& left, T_Right&& right){
        using leftElem_t = elemTOfPack_t<std::decay_t<T_Left>>;
        using rightElem_t = elemTOfPack_t<std::decay_t<T_Right>>;
        constexpr bool leftIsMask = std::is_same_v<bool, std::decay_t<leftElem_t>>;
        using pack_elem_t = std::conditional_t<leftIsMask, rightElem_t, leftElem_t>;
        static_assert(!std::is_same_v<bool, std::decay_t<pack_elem_t>>);

        /*swap operands if needed*/
        auto pack = trait::ConditionallyReverse<leftIsMask>::getFirst(std::forward<T_Left>(left), std::forward<T_Right>(right));

        auto packBeforeMul = pack;

        /*assign 0 whereever the mask is false*/
        conditionalAssignable(pack, !trait::ConditionallyReverse<leftIsMask>::getSecond(std::forward<T_Left>(left), std::forward<T_Right>(right))) = static_cast<pack_elem_t>(0);

        std::experimental::where(!trait::ConditionallyReverse<leftIsMask>::getSecond(std::forward<T_Left>(left), std::forward<T_Right>(right)), packBeforeMul) = static_cast<pack_elem_t>(0);


        const bool match = std::experimental::all_of(packBeforeMul==pack);
        if(!match){
            std::cout << "mismatch in operator*\ncorrect:    [";
            for(auto i=0u;i<laneCount_v<std::decay_t<T_Left>>;++i){
                std::cout<<packBeforeMul[i]<<" ";
            }
            std::cout << "]\ncalculated: [";
            for(auto i=0u;i<laneCount_v<std::decay_t<T_Left>>;++i){
                std::cout<<pack[i]<<" ";
            }
            std::cout << "]"<<std::endl;

            throw std::invalid_argument( "operator* was faulty" );
        }


        return pack;
    }

    BINARY_READONLY_ARITHMETIC_OP_MASK_EXCLUDED(+)
    BINARY_READONLY_ARITHMETIC_OP_MASK_EXCLUDED(-)
    BINARY_READONLY_ARITHMETIC_OP_MASK_EXCLUDED(*)
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

using alpaka::lockstep::operator+;
using alpaka::lockstep::operator-;
using alpaka::lockstep::operator*;
using alpaka::lockstep::operator/;
using alpaka::lockstep::operator&;
using alpaka::lockstep::operator&&;
using alpaka::lockstep::operator|;
using alpaka::lockstep::operator||;
using alpaka::lockstep::operator>;
using alpaka::lockstep::operator<;

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
