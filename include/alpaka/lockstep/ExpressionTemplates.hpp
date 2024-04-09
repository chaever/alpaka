///TODO Copyright, License and other stuff here

#pragma once

#include "Simd.hpp"
#include <cmath>

namespace alpaka::lockstep
{
    template<typename T_Type, typename T_Config>
    struct Variable;

    //prior to C++ 20 a static operator() is not allowed
#if __cplusplus < 202002L
    #define SIMD_EVAL_F eval
#else
    #define SIMD_EVAL_F operator()
#endif

    namespace expr
    {
        //forward declarations
        template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach, uint32_t T_dimensions>
        class BinaryXpr;
        template<typename T_Functor, typename T_Operand, typename T_Foreach, uint32_t T_dimensions>
        class UnaryXpr;
        template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
        class ReadLeafXpr;
        template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
        class WriteLeafXpr;

        template<typename T_Foreach, typename T_Elem, std::enable_if_t<!std::is_pointer_v<T_Elem>, int> = 0>
        constexpr auto load(T_Foreach const& forEach, T_Elem const & elem);

        template<typename T_Foreach, typename T_Elem>
        constexpr auto load(T_Foreach const& forEach, T_Elem const * const ptr);

        template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
        constexpr auto load(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> const& ctxVar);

        template<typename T_Foreach, typename T_Elem, std::enable_if_t<!std::is_pointer_v<T_Elem>, int> = 0>
        constexpr auto store(T_Foreach const& forEach, T_Elem & elem);

        template<typename T_Foreach, typename T_Elem>
        constexpr auto store(T_Foreach const& forEach, T_Elem * const ptr);

        template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
        constexpr auto store(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> & ctxVar);

        namespace detail
        {
            //true if T is an expression type, false otherwise.
            template<typename T>
            struct IsXpr{
                static constexpr bool value = false;
            };

            template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach, uint32_t T_dimensions>
            struct IsXpr<BinaryXpr<T_Functor, T_Left, T_Right, T_Foreach, T_dimensions>>{
                static constexpr bool value = true;
            };

            template<typename T_Functor, typename T_Operand, typename T_Foreach, uint32_t T_dimensions>
            struct IsXpr<UnaryXpr<T_Functor, T_Operand, T_Foreach, T_dimensions>>{
                static constexpr bool value = true;
            };

            template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
            struct IsXpr<ReadLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride> >{
                static constexpr bool value = true;
            };

            template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
            struct IsXpr<WriteLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride> >{
                static constexpr bool value = true;
            };

            //returns the dimensionality of an expression.
            //example: the expression that results form adding 2 vectorExpressions(each with dimensionality 1) will have dimensionality 1.
            template<typename T>
            struct GetXprDims;

            template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach, uint32_t T_dimensions>
            struct GetXprDims<BinaryXpr<T_Functor, T_Left, T_Right, T_Foreach, T_dimensions>>{
                static constexpr auto value = T_dimensions;
            };

            template<typename T_Functor, typename T_Operand, typename T_Foreach, uint32_t T_dimensions>
            struct GetXprDims<UnaryXpr<T_Functor, T_Operand, T_Foreach, T_dimensions>>{
                static constexpr auto value = T_dimensions;
            };

            template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
            struct GetXprDims<ReadLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride>>{
                static constexpr auto value = T_dimensions;
            };

            template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
            struct GetXprDims<WriteLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride>>{
                static constexpr auto value = T_dimensions;
            };

            //Default: just forward the required idx type
            template<uint32_t T_dim>
            struct DowngradeToDimensionality{
                template<typename T_Idx>
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) get(T_Idx const & idx){
                    return idx;
                }
            };

            //Do not access 0-dimensional Expressions with for example SIMD index types
            template<>
            struct DowngradeToDimensionality<0u>{
                template<typename T_Idx>
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) get(T_Idx const & idx){
                    return SingleElemIndex{};
                }
            };

            struct MaintainDimensionality{
                template<typename T_Idx>
                ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) get(T_Idx const & idx){
                    return idx;
                }
            };

        } // namespace detail

        template<typename T>
        static constexpr bool isXpr_v = detail::IsXpr<std::decay_t<T>>::value;

        template<typename T>
        static constexpr uint32_t getXprDims_v = detail::GetXprDims<std::decay_t<T>>::value;

//operations like +,-,*,/ that dont modify their operands, and whose left scalar operands need to be broadcasted
#define BINARY_READONLY_ARITHMETIC_OP(name, shortFunc)\
        struct name{\
            /*for Scalar op Scalar*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                return left shortFunc right;\
            }\
            /*for Pack op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left[0] shortFunc right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Scalar op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left    shortFunc right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(right)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::broadcast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Pack op Scalar*/\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                /*static_assert(std::is_arithmetic_v<std::decay_t<decltype(std::declval<T_Left>()[0])>>);*/\
                using result_elem_t = decltype(left[0] shortFunc right   );\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::broadcast(right);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return other;\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return other;\
            }\
        };

#define BINARY_READONLY_COMPARISON_OP(name, shortFunc)\
        struct name{\
            /*for Scalar op Scalar*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                return left shortFunc right;\
            }\
            /*for Pack op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                /*char + long int becomes long int*/\
                using result_elem_t = decltype(left[0] + right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Scalar op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left + right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(right)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::broadcast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Pack op Scalar*/\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                /*static_assert(std::is_arithmetic_v<std::decay_t<decltype(std::declval<T_Left>()[0])>>);*/\
                using result_elem_t = decltype(left[0] + right   );\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::broadcast(right);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return other;\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return other;\
            }\
        };

#define BINARY_READONLY_SHIFT_OP(name, shortFunc)\
        struct name{\
            /*for Scalar op Scalar*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                return left shortFunc right;\
            }\
            /*for Pack op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left[0] shortFunc right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Scalar op Pack*/\
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Left> && !std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left    shortFunc right[0]);\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(right)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::broadcast(left) shortFunc SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(right);\
            }\
            /*for Pack op Scalar
             * NOTE: shifts can be called with a single element (Pack_t<<4 is viable), therefore no broadcast happens below
             */\
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Left> &&  std::is_arithmetic_v<T_Right>, int> = 0 >\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                using result_elem_t = decltype(left[0] shortFunc right   );\
                using sizeIndicator_t = std::conditional_t<std::is_same_v<bool, result_elem_t>, GetSizeIndicator_t<decltype(left)>, result_elem_t>;\
                return SimdInterface_t<result_elem_t, sizeIndicator_t>::elementWiseCast(left) shortFunc right;\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
                return other;\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return alpaka::lockstep::expr::load(forEach, other);\
            }\
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
                return other;\
            }\
        };

#define UNARY_READONLY_OP_PREFIX(name, shortFunc)\
        struct name{\
            template<typename T_Operand>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Operand const& operand){\
                return shortFunc operand;\
            }\
        };

#define UNARY_READONLY_OP_POSTFIX(name, shortFunc)\
        struct name{\
            template<typename T_Operand>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Operand const& operand){\
                return operand shortFunc;\
            }\
        };

#define UNARY_FREE_FUNCTION(name, internalFunc)\
        struct name{\
            template<typename T_Operand>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Operand const& operand){\
                return internalFunc (operand);\
            }\
        };

        BINARY_READONLY_ARITHMETIC_OP(Addition, +)
        BINARY_READONLY_ARITHMETIC_OP(Subtraction, -)
        BINARY_READONLY_ARITHMETIC_OP(Multiplication, *)
        BINARY_READONLY_ARITHMETIC_OP(Division, /)
        BINARY_READONLY_ARITHMETIC_OP(BitwiseAnd, &)
        BINARY_READONLY_ARITHMETIC_OP(BitwiseOr, |)
        BINARY_READONLY_ARITHMETIC_OP(And, &&)
        BINARY_READONLY_ARITHMETIC_OP(Or, ||)

        BINARY_READONLY_COMPARISON_OP(LessThen, <)
        BINARY_READONLY_COMPARISON_OP(GreaterThen, >)

        BINARY_READONLY_SHIFT_OP(ShiftRight, >>)
        BINARY_READONLY_SHIFT_OP(ShiftLeft, <<)

        UNARY_READONLY_OP_PREFIX(BitwiseInvert, ~)
        UNARY_READONLY_OP_PREFIX(Negation, !)
        UNARY_READONLY_OP_PREFIX(PreIncrement, ++)

        UNARY_READONLY_OP_POSTFIX(PostIncrement, ++)

        UNARY_FREE_FUNCTION(Absolute, std::abs)

//clean up
#undef BINARY_READONLY_ARITHMETIC_OP
#undef BINARY_READONLY_SHIFT_OP
#undef UNARY_READONLY_OP_PREFIX
#undef UNARY_READONLY_OP_POSTFIX
#undef UNARY_FREE_FUNCTION

        struct Assignment{
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_same_v<T_Right, Pack_t<T_Left, T_Left>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const right){
                static_assert(!std::is_same_v<bool, T_Left>);
                /*std::cout << "Assignment<Pack>::operator[]: before writing " << right[0] << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/
                SimdInterface_t<T_Left, T_Left>::storeUnaligned(right, &left);
                /*std::cout << "Assignment<Pack>::operator[]: after  writing " << left     << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/
                return right;
            }
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_same_v<T_Right, Pack_t<T_Left, T_Left>>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
                /*std::cout << "Assignment<Scalar>::operator[]: before writing " << right << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/
                return left = right;
            }
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
                return expr::load(forEach, other);
            }
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
                return other;
            }
            template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){
                return expr::store(forEach, other);
            }
            template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){
                return other;
            }
        };

//needed since there is no space between "operator" and "+" in "operator+"
#define XPR_OP_WRAPPER() operator

//for operator definitions inside the BinaryXpr classes (=, +=, *= etc are not allowed as non-member functions).
//Expression must be the lefthand operand(this).
#define XPR_ASSIGN_OPERATOR\
        template<typename T_Other>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator=(T_Other const & other) const\
        {\
            auto rightXpr = Assignment::makeRightXprFromContainer(other, m_forEach);\
            constexpr auto resultDims = getXprDims_v<decltype(*this)>;\
            return BinaryXpr<Assignment, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach, resultDims>(*this, rightXpr);\
        }

//free operator definitions. Use these when possible. Expression can be the left or right operand.
#define XPR_FREE_BINARY_OPERATOR(name, shortFunc)\
        template<typename T_Left, typename T_Right, std::enable_if_t< isXpr_v<T_Left>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left left, T_Right const& right)\
        {\
            auto rightXpr = name::makeRightXprFromContainer(right, left.m_forEach);\
            constexpr auto resultDims = std::max(getXprDims_v<T_Left>, getXprDims_v<std::decay_t<decltype(rightXpr)>>);\
            return BinaryXpr<name, T_Left, decltype(rightXpr), decltype(left.m_forEach), resultDims>(left, rightXpr);\
        }\
        template<typename T_Left, typename T_Right, std::enable_if_t<!isXpr_v<T_Left> && isXpr_v<T_Right>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left const& left, T_Right right)\
        {\
            auto leftXpr = name::makeLeftXprFromContainer(left, right.m_forEach);\
            constexpr auto resultDims = std::max(getXprDims_v<T_Right>, getXprDims_v<std::decay_t<decltype(leftXpr)>>);\
            return BinaryXpr<name, decltype(leftXpr), T_Right, decltype(right.m_forEach), resultDims>(leftXpr, right);\
        }


#define XPR_FREE_UNARY_OPERATOR_PREFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Operand operand)\
        {\
            constexpr auto resultDims = getXprDims_v<T_Operand>;\
            return UnaryXpr<name, T_Operand, decltype(operand.m_forEach), resultDims>(operand);\
        }

#define XPR_FREE_UNARY_OPERATOR_POSTFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Operand operand, int neededForPrefixAndPostfixDistinction)\
        {\
            constexpr auto resultDims = getXprDims_v<T_Operand>;\
            return UnaryXpr<name, T_Operand, decltype(operand.m_forEach), resultDims>(operand);\
        }

#define XPR_UNARY_FREE_FUNCTION(name, internalFunc)\
        template<typename T_Operand, std::enable_if_t<alpaka::lockstep::expr::isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto internalFunc(T_Operand operand)\
        {\
            constexpr auto resultDims = alpaka::lockstep::expr::getXprDims_v<T_Operand>;\
            return alpaka::lockstep::expr::UnaryXpr<alpaka::lockstep::expr::name, T_Operand, decltype(operand.m_forEach), resultDims>(operand);\
        }

        XPR_FREE_BINARY_OPERATOR(Addition, +)
        XPR_FREE_BINARY_OPERATOR(Subtraction, -)
        XPR_FREE_BINARY_OPERATOR(Multiplication, *)
        XPR_FREE_BINARY_OPERATOR(Division, /)
        XPR_FREE_BINARY_OPERATOR(BitwiseAnd, &)
        XPR_FREE_BINARY_OPERATOR(BitwiseOr, |)
        XPR_FREE_BINARY_OPERATOR(ShiftLeft, <<)
        XPR_FREE_BINARY_OPERATOR(ShiftRight, >>)
        XPR_FREE_BINARY_OPERATOR(LessThen, <)
        XPR_FREE_BINARY_OPERATOR(GreaterThen, >)
        XPR_FREE_BINARY_OPERATOR(And, &&)
        XPR_FREE_BINARY_OPERATOR(Or, ||)

        XPR_FREE_UNARY_OPERATOR_PREFIX(BitwiseInvert, ~)
        XPR_FREE_UNARY_OPERATOR_PREFIX(Negation, !)
        XPR_FREE_UNARY_OPERATOR_PREFIX(PreIncrement, ++)

        XPR_FREE_UNARY_OPERATOR_POSTFIX(PostIncrement, ++)

    } // namespace expr
} // namespace alpaka::lockstep

        XPR_UNARY_FREE_FUNCTION(Absolute, std::abs)

namespace alpaka::lockstep
{
    namespace expr
    {

#undef XPR_FREE_UNARY_OPERATOR_PREFIX
#undef XPR_FREE_UNARY_OPERATOR_POSTFIX
#undef XPR_UNARY_FREE_FUNCTION

        //scalar, read-only node
        template<typename T_Foreach, typename T_Elem, uint32_t T_stride>
        class ReadLeafXpr<T_Foreach, T_Elem, 0u, T_stride>{
            ///TODO we always make a copy here, but if the value passed to the constructor is a const ref to some outside object that has the same lifetime as *this , we could save by const&
            T_Elem const m_source;
        public:
            T_Foreach const& m_forEach;

            ReadLeafXpr(T_Foreach const& forEach, T_Elem const& source) : m_source(source), m_forEach(forEach)
            {
            }

            template<uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_offset> const idx) const
            {
                return m_source;
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] SingleElemIndex const idx) const
            {
                return m_source;
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                return SimdInterface_t<T_Elem, T_Elem>::broadcast(m_source);
            }
        };

        //scalar, read-write node
        template<typename T_Foreach, typename T_Elem, uint32_t T_stride>
        class WriteLeafXpr<T_Foreach, T_Elem, 0u, T_stride>{
            T_Elem & m_dest;
        public:
            T_Foreach const& m_forEach;

            WriteLeafXpr(T_Foreach const& forEach, T_Elem & dest) : m_dest(dest), m_forEach(forEach)
            {
            }

            template<uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_offset> const idx) const
            {
                return m_dest;
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] SingleElemIndex const idx) const
            {
                return m_dest;
            }
        };

        //cannot be assigned to
        //can be made from pointers, or some container classes
        template<typename T_Foreach, typename T_Elem, uint32_t T_stride>
        class ReadLeafXpr<T_Foreach, T_Elem, 1u, T_stride>{
            T_Elem const& m_source;
        public:
            T_Foreach const& m_forEach;

            //takes a ptr that points to start of domain
            ReadLeafXpr(T_Foreach const& forEach, T_Elem const * const source) : m_source(*source), m_forEach(forEach)
            {
            }

            //allows making an expression from CtxVariable
            template<typename T_Config>
            ReadLeafXpr(T_Foreach const& forEach, typename lockstep::Variable<T_Elem, T_Config> const& v) : m_source(v[0u]), m_forEach(forEach)
            {
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                return SimdInterface_t<T_Elem, T_Elem>::loadUnaligned(&m_source + laneCount_v<T_Elem> * (m_forEach.getWorker().getWorkerIdx() + T_stride * static_cast<uint32_t>(idx)));
            }

            template<uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_offset> const idx) const
            {
                return (&m_source)[T_offset + m_forEach.getWorker().getWorkerIdx() + T_stride * static_cast<uint32_t>(idx)];
            }
        };

        //can be assigned to, and read from
        //can be made from pointers, or some container classes
        template<typename T_Foreach, typename T_Elem, uint32_t T_stride>
        class WriteLeafXpr<T_Foreach, T_Elem, 1u, T_stride>{
            T_Elem & m_dest;
        public:
            T_Foreach const& m_forEach;

            //takes a ptr that points to start of domain
            WriteLeafXpr(T_Foreach const& forEach, T_Elem * const dest) : m_dest(*dest), m_forEach(forEach)
            {
            }

            //allows making an expression from CtxVariable
            template<typename T_Config>
            WriteLeafXpr(T_Foreach const& forEach, typename lockstep::Variable<T_Elem, T_Config> & v) : m_dest(v[0u]), m_forEach(forEach)
            {
            }

            //returns ref to allow assignment
            template<uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto & operator[](ScalarLookupIndex<T_offset> const idx) const
            {
                auto const& worker = m_forEach.getWorker();
                return (&m_dest)[T_offset + worker.getWorkerIdx() + T_stride * static_cast<uint32_t>(idx)];
            }

            //returns ref to allow assignment
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto & operator[](SimdLookupIndex const idx) const
            {
                auto const& worker = m_forEach.getWorker();
                return (&m_dest)[laneCount_v<T_Elem> * (worker.getWorkerIdx() + T_stride * static_cast<uint32_t>(idx))];
            }

            XPR_ASSIGN_OPERATOR

        };

        template<typename T_Functor, typename T_Operand, typename T_Foreach, uint32_t T_dimensions>
        class UnaryXpr{
            std::conditional_t<std::is_same_v<T_Functor, Assignment>, T_Operand, const T_Operand> m_operand;
        public:
            T_Foreach const& m_forEach;

            UnaryXpr(T_Operand operand):m_operand(operand), m_forEach(operand.m_forEach)
            {
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                constexpr auto dim = getXprDims_v<T_Operand>;
                using DowngradeAsNeccessary = detail::DowngradeToDimensionality<dim>;
                return T_Functor::SIMD_EVAL_F(m_operand[DowngradeAsNeccessary::get(i)]);
            }
        };

        //const left operand, cannot assign
        template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach, uint32_t T_dimensions>
        class BinaryXpr{
            std::conditional_t<std::is_same_v<T_Functor, Assignment>, T_Left, const T_Left> m_leftOperand;
            T_Right const m_rightOperand;

            //Assignment always requests the type of element that needs to be assigned
            template<typename T_Operand>
            using optionalDowngrading = std::conditional_t<std::is_same_v<T_Functor, Assignment>, detail::MaintainDimensionality, detail::DowngradeToDimensionality<getXprDims_v<T_Operand>>>;

        public:
            T_Foreach const& m_forEach;

            BinaryXpr(T_Left left, T_Right right):m_leftOperand(left), m_rightOperand(right), m_forEach(left.m_forEach)
            {
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                using DowngradeLeft  = optionalDowngrading<T_Left>;
                using DowngradeRight = optionalDowngrading<T_Right>;
                return T_Functor::SIMD_EVAL_F(m_leftOperand[DowngradeLeft::get(i)], m_rightOperand[DowngradeRight::get(i)]);
            }

            XPR_ASSIGN_OPERATOR
        };

        ///TODO T_Elem should maybe also be deduced?
        template<typename T_Elem, typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluateExpression(T_Xpr const& xpr)
        {
            constexpr auto lanes = laneCount_v<T_Elem>;
            constexpr auto numWorkers = std::decay_t<decltype(xpr.m_forEach.getWorker())>::numWorkers;
            constexpr auto domainSize = std::decay_t<decltype(xpr.m_forEach)>::domainSize;

            constexpr auto simdLoops = domainSize/(numWorkers*lanes);

            constexpr auto elementsProcessedBySimd = simdLoops*lanes*numWorkers;

            const auto workerIdx = xpr.m_forEach.getWorker().getWorkerIdx();

            //std::cout << "evaluateExpression: running " << simdLoops << " simdLoops and " << (domainSize - simdLoops*lanes*numWorkers) << " scalar loops." << std::endl;

            for(uint32_t i = 0u; i<simdLoops; ++i){
                //std::cout << "evaluateExpression: starting vectorLoop " << i << std::endl;
                //uses the operator[] that returns const Pack_t
                xpr[SimdLookupIndex{i}];
                //std::cout << "evaluateExpression: finished vectorLoop " << i << std::endl;
            }
            for(uint32_t i = 0u; i<(domainSize-elementsProcessedBySimd); ++i){
                //std::cout << "evaluateExpression: starting scalarLoop " << i << std::endl;
                //uses the operator[] that returns const T_Elem &
                xpr[ScalarLookupIndex<elementsProcessedBySimd>{i}];
            }
        }

        template<typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto evalToCtxVar(T_Xpr const& xpr){

            using Elem_t = std::decay_t<decltype(xpr[std::declval<ScalarLookupIndex<0u>>()])>;
            auto const& forEach = xpr.m_forEach;
            using ContextVar_t = Variable<Elem_t, typename std::decay_t<decltype(forEach)>::BaseConfig>;

            ContextVar_t tmp;
            evaluateExpression<Elem_t>(store(forEach, tmp) = xpr);
            return tmp;
        }

        //single element, broadcasted if required
        template<typename T_Foreach, typename T_Elem, std::enable_if_t<!std::is_pointer_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Foreach const& forEach, T_Elem const & elem){
            return ReadLeafXpr<T_Foreach, T_Elem, 0u, 0u>(forEach, elem);
        }

        //pointer to threadblocks data
        template<typename T_Foreach, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Foreach const& forEach, T_Elem const * const ptr){
            constexpr uint32_t stride = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
            return ReadLeafXpr<T_Foreach, T_Elem, 1u, stride>(forEach, ptr);
        }

        //lockstep ctxVar
        template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> const& ctxVar){
            constexpr uint32_t stride = 1u;
            return ReadLeafXpr<T_Foreach<T_Worker, T_Config>, T_Elem, 1u, stride>(forEach, ctxVar);
        }

        template<typename T_Foreach, typename T_Elem, std::enable_if_t<!std::is_pointer_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Foreach const& forEach, T_Elem & elem){
            return WriteLeafXpr<T_Foreach, T_Elem, 0u, 0u>(forEach, elem);
        }

        template<typename T_Foreach, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Foreach const& forEach, T_Elem * const ptr){
            constexpr uint32_t stride = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
            return WriteLeafXpr<T_Foreach, T_Elem, 1u, stride>(forEach, ptr);
        }

        template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> & ctxVar){
            constexpr uint32_t stride = 1u;
            return WriteLeafXpr<T_Foreach<T_Worker, T_Config>, T_Elem, 1u, stride>(forEach, ctxVar);
        }

//clean up
#undef XPR_OP_WRAPPER
#undef XPR_ASSIGN_OPERATOR

    } // namespace expr
} // namespace alpaka::lockstep
