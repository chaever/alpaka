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
        namespace dataLocationTags{
            class scalar{};
            template<typename T_Config>
            class ctxVar{};
            class perBlockArray{};
        }

        //forward declarations
        template<typename T_Functor, typename T_Left, typename T_Right>
        class BinaryXpr;
        template<typename T_Functor, typename T_Operand>
        class UnaryXpr;
        template<typename T_Elem, typename dataLocationTag>
        class ReadLeafXpr;
        template<typename T_Elem, typename dataLocationTag>
        class WriteLeafXpr;

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Elem const & elem);

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Elem const * const ptr);

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(lockstep::Variable<T_Elem, T_Config> const& ctxVar);

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem & elem);

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem * const ptr);

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(lockstep::Variable<T_Elem, T_Config> & ctxVar);

        namespace detail
        {
            //true if T is an expression type, false otherwise.
            template<typename T>
            struct IsXpr{
                static constexpr bool value = false;
            };

            template<typename T_Functor, typename T_Left, typename T_Right>
            struct IsXpr<BinaryXpr<T_Functor, T_Left, T_Right>>{
                static constexpr bool value = true;
            };

            template<typename T_Functor, typename T_Operand>
            struct IsXpr<UnaryXpr<T_Functor, T_Operand>>{
                static constexpr bool value = true;
            };

            template<typename T_Elem, typename dataLocationTag>
            struct IsXpr<ReadLeafXpr<T_Elem, dataLocationTag> >{
                static constexpr bool value = true;
            };

            template<typename T_Elem, typename dataLocationTag>
            struct IsXpr<WriteLeafXpr<T_Elem, dataLocationTag> >{
                static constexpr bool value = true;
            };

            template<typename T_Elem, typename T_TypeToWrite>
            struct AssignmentDestination{
                T_Elem & dest;
            };

        } // namespace detail

        template<typename T>
        static constexpr bool isXpr_v = detail::IsXpr<std::decay_t<T>>::value;

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
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return other;\
            }\
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
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
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return other;\
            }\
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
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
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){\
                return other;\
            }\
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
                return alpaka::lockstep::expr::load(other);\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){\
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
            /*Pack op Pack*/
            template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_arithmetic_v<T_Right>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>> left, T_Right const right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const castedPack = SimdInterface_t<T_Left, T_Left>::elementWiseCast(right);
                SimdInterface_t<T_Left, T_Left>::storeUnaligned(castedPack, &left.dest);
                return castedPack;
            }
            /*Scalar op Scalar*/
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Right>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, T_Left> left, T_Right const& right){
                return left.dest = right;
            }
            /*Pack op Scalar*/
            template<typename T_Left, typename T_Right, std::enable_if_t< std::is_arithmetic_v<T_Right>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>> left, T_Right const& right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const expandedValue = SimdInterface_t<T_Left, T_Left>::broadcast(right);
                SimdInterface_t<T_Left, T_Left>::storeUnaligned(expandedValue, &left.dest);
                return expandedValue;
            }
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){
                return expr::load(other);
            }
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other){
                return other;
            }
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){
                return expr::store(other);
            }
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other){
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
            auto rightXpr = Assignment::makeRightXprFromContainer(other);\
            return BinaryXpr<Assignment, std::decay_t<decltype(*this)>, std::decay_t<decltype(rightXpr)>>(*this, rightXpr);\
        }

//free operator definitions. Use these when possible. Expression can be the left or right operand.
#define XPR_FREE_BINARY_OPERATOR(name, shortFunc)\
        template<typename T_Left, typename T_Right, std::enable_if_t< isXpr_v<T_Left>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left left, T_Right const& right)\
        {\
            auto rightXpr = name::makeRightXprFromContainer(right);\
            return BinaryXpr<name, T_Left, std::decay_t<decltype(rightXpr)>>(left, rightXpr);\
        }\
        template<typename T_Left, typename T_Right, std::enable_if_t<!isXpr_v<T_Left> && isXpr_v<T_Right>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left const& left, T_Right right)\
        {\
            auto leftXpr = name::makeLeftXprFromContainer(left);\
            return BinaryXpr<name, decltype(leftXpr), T_Right>(leftXpr, right);\
        }


#define XPR_FREE_UNARY_OPERATOR_PREFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Operand operand)\
        {\
            return UnaryXpr<name, T_Operand>(operand);\
        }

#define XPR_FREE_UNARY_OPERATOR_POSTFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Operand operand, int neededForPrefixAndPostfixDistinction)\
        {\
            return UnaryXpr<name, T_Operand>(operand);\
        }

#define XPR_UNARY_FREE_FUNCTION(name, internalFunc)\
        template<typename T_Operand, std::enable_if_t<alpaka::lockstep::expr::isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto internalFunc(T_Operand operand)\
        {\
            return alpaka::lockstep::expr::UnaryXpr<alpaka::lockstep::expr::name, T_Operand>(operand);\
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
        template<typename T_Elem>
        class ReadLeafXpr<T_Elem, dataLocationTags::scalar>{
            ///TODO we always make a copy here, but if the value passed to the constructor is a const ref to some outside object that has the same lifetime as *this , we could save by const&
            T_Elem const m_source;
        public:

            ReadLeafXpr(T_Elem const& source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                return m_source;
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                return m_source;
            }
        };

        //scalar, write-only node
        template<typename T_Elem>
        class WriteLeafXpr<T_Elem, dataLocationTags::scalar>{
            T_Elem & m_dest;
        public:

            WriteLeafXpr(T_Elem & dest) : m_dest(dest)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                return detail::AssignmentDestination<T_Elem, T_Elem>{m_dest};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                return detail::AssignmentDestination<T_Elem, T_Elem>{m_dest};
            }

            XPR_ASSIGN_OPERATOR
        };

        //cannot be assigned to
        //can be made from pointers
        template<typename T_Elem>
        class ReadLeafXpr<T_Elem, dataLocationTags::perBlockArray>{
            T_Elem const * const m_source;
        public:

            //takes a ptr that points to start of domain
            ReadLeafXpr(T_Elem const * const source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                const auto & worker = idx.m_forEach.getWorker();
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx);
                //std::cout << "ReadLeafXpr<dataLocationTags::perBlockArray>: ScalarLookupIndex<T_offset=" << T_offset << ">(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return m_source[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)];
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                const auto & worker = idx.m_forEach.getWorker();
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx));
                //std::cout << "ReadLeafXpr<dataLocationTags::perBlockArray>: SimdLookupIndex(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return SimdInterface_t<T_Elem, T_Elem>::loadUnaligned(m_source + laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)));
            }
        };

        //cannot be assigned to
        template<typename T_Elem, typename T_Config>
        class ReadLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>{
            lockstep::Variable<T_Elem, T_Config> const& m_source;
        public:

            ReadLeafXpr(lockstep::Variable<T_Elem, T_Config> const& source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = T_offset + static_cast<uint32_t>(idx);
                //std::cout << "ReadLeafXpr<dataLocationTags::ctxVar>: ScalarLookupIndex<T_offset=" << T_offset << ">(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << ", value=" << m_source[offset] << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return m_source[T_offset + static_cast<uint32_t>(idx)];
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = laneCount_v<T_Elem>*static_cast<uint32_t>(idx);
                //std::cout << "ReadLeafXpr<dataLocationTags::ctxVar>: SimdLookupIndex(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return SimdInterface_t<T_Elem, T_Elem>::loadUnaligned(&(m_source[laneCount_v<T_Elem>*static_cast<uint32_t>(idx)]));
            }
        };


        //can be assigned to
        //can be made from pointers, or some container classes
        template<typename T_Elem>
        class WriteLeafXpr<T_Elem, dataLocationTags::perBlockArray>{
            T_Elem * const m_dest;
        public:

            WriteLeafXpr(T_Elem * const dest) : m_dest(dest)
            {
            }
            //returns ref to allow assignment
            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = T_offset + static_cast<uint32_t>(idx);
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>: ScalarLookupIndex<T_offset=" << T_offset << ">(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return detail::AssignmentDestination<T_Elem, T_Elem>{m_dest[T_offset + static_cast<uint32_t>(idx)]};
            }

            //returns ref to allow assignment
            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                auto const& worker = idx.m_forEach.getWorker();
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx));
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>: SimdLookupIndex(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return detail::AssignmentDestination<T_Elem, Pack_t<T_Elem, T_Elem>>{m_dest[laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))]};
            }

            XPR_ASSIGN_OPERATOR

        };

        //can be assigned to
        template<typename T_Elem, typename T_Config>
        class WriteLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>{
            lockstep::Variable<T_Elem, T_Config> & m_dest;
        public:

            WriteLeafXpr(lockstep::Variable<T_Elem, T_Config> & dest) : m_dest(dest)
            {
            }

            //returns ref to allow assignment
            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = T_offset + static_cast<uint32_t>(idx);
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>: ScalarLookupIndex<T_offset=" << T_offset << ">(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return detail::AssignmentDestination<T_Elem, T_Elem>{m_dest[T_offset + static_cast<uint32_t>(idx)]};
            }

            //returns ref to allow assignment
            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                auto offset = laneCount_v<T_Elem>*static_cast<uint32_t>(idx);
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>: SimdLookupIndex(idx=" << static_cast<uint32_t>(idx) << ") -> offset=" << offset << std::endl;
                //ALPAKA_ASSERT_ACC(offset < 11 && offset >= 0);
#endif
                return detail::AssignmentDestination<T_Elem, Pack_t<T_Elem, T_Elem>>{m_dest[laneCount_v<T_Elem>*static_cast<uint32_t>(idx)]};
            }

            XPR_ASSIGN_OPERATOR

        };

        template<typename T_Functor, typename T_Operand>
        class UnaryXpr{
            const T_Operand m_operand;
        public:

            UnaryXpr(T_Operand operand):m_operand(operand)
            {
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                return T_Functor::SIMD_EVAL_F(m_operand[i]);
            }
        };

        //const left operand, cannot assign
        template<typename T_Functor, typename T_Left, typename T_Right>
        class BinaryXpr{
            std::conditional_t<std::is_same_v<T_Functor, Assignment>, T_Left, const T_Left> m_leftOperand;
            T_Right const m_rightOperand;

        public:

            BinaryXpr(T_Left left, T_Right right):m_leftOperand(left), m_rightOperand(right)
            {
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
            }

            XPR_ASSIGN_OPERATOR
        };

        template<typename T_Foreach, typename T_Elem, typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluateExpression(T_Foreach const& forEach, T_Xpr const& xpr)
        {
            constexpr auto lanes = laneCount_v<T_Elem>;
            constexpr auto numWorkers = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
            constexpr auto domainSize = std::decay_t<decltype(forEach)>::domainSize;

            constexpr auto simdLoops = domainSize/(numWorkers*lanes);

            constexpr auto elementsProcessedBySimd = simdLoops*numWorkers*lanes;

            constexpr auto leftovers = domainSize - elementsProcessedBySimd;

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
            //std::cout << "evaluateExpression: running " << simdLoops << " simdLoops and " << (domainSize - elementsProcessedBySimd) << " scalar loops. numWorkers=" << numWorkers << ", domainSize=" << domainSize << ", laneCount=" << lanes << std::endl;
#endif

            for(uint32_t i = 0u; i<simdLoops; ++i){
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                std::cout << "evaluateExpression: Worker "<<forEach.getWorker().getWorkerIdx()<<" starting vectorLoop " << i << std::endl;
#endif
                //uses the operator[] that returns const Pack_t
                xpr[SimdLookupIndex{forEach, i}];
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                std::cout << "evaluateExpression: Worker "<<forEach.getWorker().getWorkerIdx()<<" finished vectorLoop " << i << std::endl;
#endif
            }

            for(uint32_t i = 0u; i<leftovers; i+=numWorkers){
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                std::cout << "evaluateExpression: Worker "<<forEach.getWorker().getWorkerIdx()<<" starting scalarLoop " << i << std::endl;
#endif
                //uses the operator[] that returns const T_Elem &
                xpr[ScalarLookupIndex<T_Foreach, elementsProcessedBySimd>{forEach, i}];
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
                std::cout << "evaluateExpression: Worker "<<forEach.getWorker().getWorkerIdx()<<" finished scalarLoop " << i << std::endl;
#endif
            }
#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED
            //std::cout << "evaluateExpression: returning." << std::endl;
#endif
        }

        template<typename T_Foreach, typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto evalToCtxVar(T_Foreach const& forEach, T_Xpr const& xpr){

            using Elem_t = std::decay_t<decltype(xpr[std::declval<ScalarLookupIndex<T_Foreach, 0u>>()])>;
            using ContextVar_t = Variable<Elem_t, typename std::decay_t<decltype(forEach)>::BaseConfig>;

            ContextVar_t tmp;
            auto storeXpr = (store(tmp) = xpr);
            evaluateExpression<T_Foreach, Elem_t, decltype(storeXpr)>(forEach, storeXpr);
            return tmp;
        }

        //single element, broadcasted if required
        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Elem const & elem){
            return ReadLeafXpr<T_Elem, dataLocationTags::scalar>(elem);
        }

        //pointer to threadblocks data
        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Elem const * const ptr){
            return ReadLeafXpr<T_Elem, dataLocationTags::perBlockArray>(ptr);
        }

        //lockstep ctxVar
        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(lockstep::Variable<T_Elem, T_Config> const& ctxVar){
            return ReadLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>(ctxVar);
        }

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem & elem){
            return WriteLeafXpr<T_Elem, dataLocationTags::scalar>(elem);
        }

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem * const ptr){
            return WriteLeafXpr<T_Elem, dataLocationTags::perBlockArray>(ptr);
        }

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(lockstep::Variable<T_Elem, T_Config> & ctxVar){
            return WriteLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>(ctxVar);
        }

//clean up
#undef XPR_OP_WRAPPER
#undef XPR_ASSIGN_OPERATOR

    } // namespace expr
} // namespace alpaka::lockstep
