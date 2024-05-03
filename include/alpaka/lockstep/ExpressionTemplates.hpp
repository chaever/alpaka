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
        template<typename T_Lambda>
        using MemAccessorFunctor = alpaka::lockstep::MemAccessorFunctor<T_Lambda>;

        namespace dataLocationTags{
            class scalar{};
            template<typename T_Config>
            class ctxVar{};
            class perBlockArray{};
            template<typename T_Lambda>
            class gatherScatterFunctor{};
        }

        namespace memoryLayouts{
            class contigous{};
            template<typename T_Lambda>
            class nonContigous{};
        }

        //forward declarations
        template<typename T_Functor, typename T_Left, typename T_Right>
        class BinaryXpr;
        template<typename T_Functor, typename T_Operand>
        class UnaryXpr;
        template<typename T_Lambda, typename... T_Args>
        class NAryXpr;
        template<typename T_Elem, typename dataLocationTag>
        class ReadLeafXpr;
        template<typename T_Elem, typename dataLocationTag>
        class WriteLeafXpr;

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T_Elem>>, int> = 0 >
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(T_Elem&& elem);

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(T_Elem const * const ptr);

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(lockstep::Variable<T_Elem, T_Config> const& ctxVar);

        template<typename T_Lambda>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(MemAccessorFunctor<T_Lambda> func);

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> = 0>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem & elem);

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem * const ptr);

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(lockstep::Variable<T_Elem, T_Config> & ctxVar);

        template<typename T_Lambda>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(MemAccessorFunctor<T_Lambda> func);

        namespace trait{

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

            template<typename T_Lambda, typename... T_Args>
            struct IsXpr<NAryXpr<T_Lambda, T_Args...>>{
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

            template<typename T_Xpr>
            struct AllChildrenAreConst;

            template<typename T_Functor, typename T_Left, typename T_Right>
            struct AllChildrenAreConst<BinaryXpr<T_Functor, T_Left, T_Right>>{
                static constexpr bool leftIsConst = AllChildrenAreConst<std::decay_t<T_Left>>::value;
                static constexpr bool rightIsConst = AllChildrenAreConst<std::decay_t<T_Right>>::value;
                static constexpr bool value = leftIsConst&&rightIsConst;
            };

            template<typename T_Functor, typename T_Operand>
            struct AllChildrenAreConst<UnaryXpr<T_Functor, T_Operand>>{
                static constexpr bool value = AllChildrenAreConst<std::decay_t<T_Operand>>::value;
            };

            template<typename T_Lambda, typename... T_Args>
            struct AllChildrenAreConst<NAryXpr<T_Lambda, T_Args...>>{
                static constexpr bool value = std::conjunction_v<AllChildrenAreConst<std::decay_t<T_Args>>...>;
            };

            template<typename T_Elem, typename dataLocationTag>
            struct AllChildrenAreConst<ReadLeafXpr<T_Elem, dataLocationTag> >{
                static constexpr bool value = true;
            };

            template<typename T_Elem, typename dataLocationTag>
            struct AllChildrenAreConst<WriteLeafXpr<T_Elem, dataLocationTag> >{
                static constexpr bool value = false;
            };
        }

        namespace detail
        {

            template<typename T_Elem, typename T_TypeToWrite, typename T_memLayout>
            struct AssignmentDestination;

            template<typename T_Elem, typename T_TypeToWrite>
            struct AssignmentDestination<T_Elem, T_TypeToWrite, memoryLayouts::contigous>{
                T_Elem & dest;
                static_assert(!std::is_const_v<T_Elem>);
                static_assert(!std::is_reference_v<T_Elem>);
                constexpr AssignmentDestination(T_Elem & elem):dest(elem)
                {
                }
            };

            template<typename T_Elem, typename T_TypeToWrite, typename T_Lambda>
            struct AssignmentDestination<T_Elem, T_TypeToWrite, memoryLayouts::nonContigous<T_Lambda>>{
                MemAccessorFunctor<T_Lambda> const& storeFunc;
                uint32_t const offset;
                static_assert(!std::is_const_v<T_Elem>);
                static_assert(!std::is_reference_v<T_Elem>);
                constexpr AssignmentDestination(MemAccessorFunctor<T_Lambda> const& func, uint32_t const offset):storeFunc(func), offset(offset)
                {
                }
            };

        } // namespace detail

        template<typename T>
        static constexpr bool allChildrenAreConst_v = trait::AllChildrenAreConst<std::decay_t<T>>::value;

        template<typename T>
        static constexpr bool isXpr_v = trait::IsXpr<std::decay_t<T>>::value;

#define BINARY_READONLY_OP(name, shortFunc)\
        struct name{\
            template<typename T_Left, typename T_Right>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
                return left shortFunc right;\
            }\
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other&& other){\
                return load(std::forward<T_Other>(other));\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other&& other){\
                return std::forward<T_Other>(other);\
            }\
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other&& other){\
                return load(std::forward<T_Other>(other));\
            }\
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other&& other){\
                return std::forward<T_Other>(other);\
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

        BINARY_READONLY_OP(Addition, +)
        BINARY_READONLY_OP(Subtraction, -)
        BINARY_READONLY_OP(Multiplication, *)
        BINARY_READONLY_OP(Division, /)
        BINARY_READONLY_OP(BitwiseAnd, &)
        BINARY_READONLY_OP(BitwiseOr, |)
        BINARY_READONLY_OP(And, &&)
        BINARY_READONLY_OP(Or, ||)
        BINARY_READONLY_OP(LessThen, <)
        BINARY_READONLY_OP(GreaterThen, >)
        BINARY_READONLY_OP(ShiftRight, >>)
        BINARY_READONLY_OP(ShiftLeft, <<)

        UNARY_READONLY_OP_PREFIX(BitwiseInvert, ~)
        UNARY_READONLY_OP_PREFIX(Negation, !)
        UNARY_READONLY_OP_PREFIX(PreIncrement, ++)

        UNARY_READONLY_OP_POSTFIX(PostIncrement, ++)

        UNARY_FREE_FUNCTION(Absolute, std::abs)

//clean up
#undef BINARY_READONLY_OP
#undef UNARY_READONLY_OP_PREFIX
#undef UNARY_READONLY_OP_POSTFIX
#undef UNARY_FREE_FUNCTION

        struct Assignment{
            /*Pack op Pack*/
            template<typename T_Left, typename T_Right, std::enable_if_t<isPackWrapper_v<T_Right> && (T_Right::laneCount>1), int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>, memoryLayouts::contigous> left, T_Right const right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const castedPack = Pack_t<T_Left, T_Left>(right);

                //std::cout << "Assignment::eval(SimdPack): before store to address " << reinterpret_cast<uint64_t>(&left.dest) << std::endl;
                Pack_t<T_Left, T_Left>::storeUnaligned(castedPack, &left.dest);
                //std::cout << "Assignment::eval(SimdPack): after  store" << std::endl;
                return castedPack;
            }
            /*Pack op Pack, nonContigous*/
            template<typename T_Left, typename T_Right, typename T_Lambda, std::enable_if_t<isPackWrapper_v<T_Right> && (T_Right::laneCount>1), int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>, memoryLayouts::nonContigous<T_Lambda>> left, T_Right const right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const castedPack = Pack_t<T_Left, T_Left>(right);

                for(auto i=0u;i<laneCount_v<T_Left>;++i){
                    left.storeFunc[left.offset+i] = castedPack[i];
                    //std::cout << "Assignment::eval<Pack>: idx="<<(left.offset+i)<<std::endl;
                }

                return castedPack;
            }
            /*Scalar op Scalar*/
            template<typename T_Left, typename T_Right, std::enable_if_t<isPackWrapper_v<T_Right> && T_Right::laneCount==1, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, T_Left, memoryLayouts::contigous> left, T_Right const& right){
                return left.dest = right;
            }
            /*Scalar op Scalar, nonContigous*/
            template<typename T_Left, typename T_Right, typename T_Lambda, std::enable_if_t<isPackWrapper_v<T_Right> && T_Right::laneCount==1, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, T_Left, memoryLayouts::nonContigous<T_Lambda>> left, T_Right const& right){
                return left.storeFunc[left.offset] = static_cast<T_Left>(right.packContent);
            }
            /*Pack op Scalar*/
            template<typename T_Left, typename T_Right, std::enable_if_t<isPackWrapper_v<T_Right> && T_Right::laneCount==1, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>, memoryLayouts::contigous> left, T_Right const& right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const expandedValue = Pack_t<T_Left, T_Left>(right);
                Pack_t<T_Left, T_Left>::storeUnaligned(expandedValue, &left.dest);
                //return single value
                return static_cast<T_Left>(right);
            }
            /*Pack op Scalar, nonContigous*/
            template<typename T_Left, typename T_Right, typename T_Lambda, std::enable_if_t<isPackWrapper_v<T_Right> && T_Right::laneCount==1, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(detail::AssignmentDestination<T_Left, Pack_t<T_Left, T_Left>, memoryLayouts::nonContigous<T_Lambda>> left, T_Right const& right){
                static_assert(!std::is_same_v<bool, T_Left>);
                auto const castedValue = static_cast<T_Left>(right);
                for(auto i=0u;i<laneCount_v<T_Left>;++i){
                    left.storeFunc[left.offset+i] = castedValue;
                }
                return Pack_t<T_Left, T_Left>::broadcast(right);
            }

            ///TODO missing:
            //Pack op OneElemPack
            //scalar op OneElemPack




            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other&& other){
                return expr::load(std::forward<T_Other>(other));
            }
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other&& other){
                return std::forward<T_Other>(other);
            }
            template<typename T_Other, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other&& other){
                return expr::store(std::forward<T_Other>(other));
            }
            template<typename T_Other, std::enable_if_t< isXpr_v<T_Other>, int> = 0>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other&& other){
                return std::forward<T_Other>(other);
            }
        };

//needed since there is no space between "operator" and "+" in "operator+"
#define XPR_OP_WRAPPER() operator

//for operator definitions inside the BinaryXpr classes (=, +=, *= etc are not allowed as non-member functions).
//Expression must be the lefthand operand(this).
#define XPR_ASSIGN_OPERATOR\
        template<typename T_Other>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator=(T_Other&& other)\
        {\
            decltype(auto) rightXpr = Assignment::makeRightXprFromContainer(std::forward<T_Other>(other));\
            /*static_assert(std::is_const_v<std::remove_reference_t<decltype(rightXpr)>>);*/\
            return BinaryXpr<Assignment, std::remove_reference_t<decltype(*this)>, std::remove_reference_t<decltype(rightXpr)>>::makeConstIfPossible(*this, rightXpr);\
        }

//free operator definitions. Use these when possible. Expression can be the left or right operand.
#define XPR_FREE_BINARY_OPERATOR(name, shortFunc)\
        template<typename T_Left, typename T_Right, std::enable_if_t< isXpr_v<T_Left>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()shortFunc(T_Left&& left, T_Right&& right)\
        {\
            decltype(auto) rightXpr = name::makeRightXprFromContainer(std::forward<T_Right>(right));\
            return BinaryXpr<name, std::remove_reference_t<T_Left>, std::remove_reference_t<decltype(rightXpr)>>::makeConstIfPossible(std::forward<T_Left>(left), rightXpr);\
        }\
        template<typename T_Left, typename T_Right, std::enable_if_t<!isXpr_v<T_Left> && isXpr_v<T_Right>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()shortFunc(T_Left&& left, T_Right&& right)\
        {\
            decltype(auto) leftXpr = name::makeLeftXprFromContainer(std::forward<T_Left>(left));\
            return BinaryXpr<name, std::remove_reference_t<decltype(leftXpr)>, std::remove_reference_t<T_Right>>::makeConstIfPossible(leftXpr, std::forward<T_Right>(right));\
        }

#define XPR_FREE_UNARY_OPERATOR_PREFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()shortFunc(T_Operand&& operand)\
        {\
            return UnaryXpr<name, std::remove_reference_t<T_Operand>>::makeConstIfPossible(std::forward<T_Operand>(operand));\
        }

#define XPR_FREE_UNARY_OPERATOR_POSTFIX(name, shortFunc)\
        template<typename T_Operand, std::enable_if_t<isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) XPR_OP_WRAPPER()shortFunc(T_Operand&& operand, [[unused]] int neededForPrefixAndPostfixDistinction)\
        {\
            return UnaryXpr<name, std::remove_reference_t<T_Operand>>::makeConstIfPossible(std::forward<T_Operand>(operand));\
        }

#define XPR_UNARY_FREE_FUNCTION(name, internalFunc)\
        template<typename T_Operand, std::enable_if_t<alpaka::lockstep::expr::isXpr_v<T_Operand>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) internalFunc(T_Operand&& operand)\
        {\
            return alpaka::lockstep::expr::UnaryXpr<alpaka::lockstep::expr::name, std::remove_reference_t<T_Operand>>::makeConstIfPossible(std::forward<T_Operand>(operand));\
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

#undef XPR_FREE_BINARY_OPERATOR
#undef XPR_FREE_UNARY_OPERATOR_PREFIX
#undef XPR_FREE_UNARY_OPERATOR_POSTFIX
#undef XPR_UNARY_FREE_FUNCTION

        //scalar, read-only node
        template<typename T_Elem>
        class ReadLeafXpr<T_Elem, dataLocationTags::scalar>{
            ///TODO we always make a copy here, but if the value passed to the constructor is a const ref to some outside object that has the same lifetime as *this , we could save by const&
            T_Elem const m_source;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ReadLeafXpr(T_Elem const& source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const) const
            {
                return OneElemPack_t<T_Elem, T_Elem>{m_source};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const) const
            {
                return OneElemPack_t<T_Elem, T_Elem>{m_source};
            }
        };

        //scalar, write-only node
        template<typename T_Elem>
        class WriteLeafXpr<T_Elem, dataLocationTags::scalar>{
            T_Elem & m_dest;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr WriteLeafXpr(T_Elem & dest) : m_dest(dest)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const)
            {
                return detail::AssignmentDestination<T_Elem, T_Elem, memoryLayouts::contigous>{m_dest};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const)
            {
                return detail::AssignmentDestination<T_Elem, T_Elem, memoryLayouts::contigous>{m_dest};
            }

            XPR_ASSIGN_OPERATOR
        };

        //cannot be assigned to
        //can be made from pointers
        template<typename T_Elem>
        class ReadLeafXpr<T_Elem, dataLocationTags::perBlockArray>{
            T_Elem const * const m_source;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            //takes a ptr that points to start of domain
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ReadLeafXpr(T_Elem const * const source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                const auto & worker = idx.m_forEach.getWorker();

                //std::cout << "ReadLeafXpr <dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): accessing index " << T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): value is: " << m_source[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)] << std::endl;

                return OneElemPack_t<T_Elem, T_Elem>{m_source[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)]};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                const auto & worker = idx.m_forEach.getWorker();

                //std::cout << "ReadLeafXpr <dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): accessing index " << laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): value is: " << m_source[laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))] << std::endl;
                //printf("ReadLeafXpr<perBlockArray> thread %d in block %d: before load\n", threadIdx.x, blockIdx.x);
                //auto tmp = SimdInterface_t<T_Elem, T_Elem>::loadUnaligned(m_source + laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)));
                //printf("ReadLeafXpr<perBlockArray> thread %d in block %d: after  load, value is %d\n", threadIdx.x, blockIdx.x, tmp);

                return Pack_t<T_Elem, T_Elem>::loadUnaligned(m_source + laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)));
            }
        };

        //cannot be assigned to
        template<typename T_Elem, typename T_Config>
        class ReadLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>{
            lockstep::Variable<T_Elem, T_Config> const& m_source;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ReadLeafXpr(lockstep::Variable<T_Elem, T_Config> const& source) : m_source(source)
            {
                //printf("ReadLeafXpr<ctxVar> thread %d in block %d: in CTOR\n", threadIdx.x, blockIdx.x);
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::Constructor(ctxVar): object at " << reinterpret_cast<uint64_t>(this) << " has &ctxVar=" << reinterpret_cast<uint64_t>(&source) << std::endl;
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): accessing index " << T_offset + static_cast<uint32_t>(idx) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): value is: " << m_source[T_offset + static_cast<uint32_t>(idx)] << std::endl;
                return OneElemPack_t<T_Elem, T_Elem>{m_source[T_offset + static_cast<uint32_t>(idx)]};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);

                //const auto offset = laneCount_v<T_Elem>*static_cast<uint32_t>(idx);
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): accessing index " << offset << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): &ctxVar[0]=" << reinterpret_cast<uint64_t>(&m_source[0]) << ", &ctxVar[offset]=" << reinterpret_cast<uint64_t>(&m_source[offset]) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): &ctxVar=" << reinterpret_cast<uint64_t>(&m_source) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): value is: " << m_source[offset] << std::endl;
                //printf("ReadLeafXpr<ctxVar> thread %d in block %d: before load\n", threadIdx.x, blockIdx.x);
                //auto tmp = SimdInterface_t<T_Elem, T_Elem>::loadUnaligned(&(m_source[offset]));
                //printf("ReadLeafXpr<ctxVar> thread %d in block %d: after  load, value is %d\n", threadIdx.x, blockIdx.x, tmp);

                return Pack_t<T_Elem, T_Elem>::loadUnaligned(&(m_source[laneCount_v<T_Elem>*static_cast<uint32_t>(idx)]));
            }
        };

        //can be assigned to
        //can be made from pointers, or some container classes
        template<typename T_Elem>
        class WriteLeafXpr<T_Elem, dataLocationTags::perBlockArray>{

            T_Elem * const m_dest;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr WriteLeafXpr(T_Elem * const dest) : m_dest(dest)
            {
            }
            //returns ref to allow assignment
            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx)
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                auto const& worker = idx.m_forEach.getWorker();
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): accessing index " << T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx) << std::endl;
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): address is: " << reinterpret_cast<uint64_t>(&m_dest[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)]) << std::endl;
                return detail::AssignmentDestination<T_Elem, T_Elem, memoryLayouts::contigous>{m_dest[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)]};
            }

            //returns ref to allow assignment
            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx)
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                auto const& worker = idx.m_forEach.getWorker();

                const auto offset = laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx));
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): accessing index " << offset << std::endl;
                //std::cout << "WriteLeafXpr<dataLocationTags::perBlockArray>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): accessed address is: " << reinterpret_cast<uint64_t>(&m_dest[offset]) << std::endl;
                //printf("WriteLeafXpr<perBlockArray> thread %d in block %d: before load\n", threadIdx.x, blockIdx.x);
                //auto tmp = m_dest[offset];
                //printf("WriteLeafXpr<perBlockArray> thread %d in block %d: after  load, value is %d\n", threadIdx.x, blockIdx.x, tmp);

                return detail::AssignmentDestination<T_Elem, Pack_t<T_Elem, T_Elem>, memoryLayouts::contigous>{m_dest[laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))]};
            }

            XPR_ASSIGN_OPERATOR

        };

        //can be assigned to
        template<typename T_Elem, typename T_Config>
        class WriteLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>{
            lockstep::Variable<T_Elem, T_Config> & m_dest;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr WriteLeafXpr(lockstep::Variable<T_Elem, T_Config> & dest) : m_dest(dest)
            {
            }

            //returns ref to allow assignment
            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx)
            {
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): accessing index " << T_offset + static_cast<uint32_t>(idx) << std::endl;
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): address is: " << reinterpret_cast<uint64_t>(&m_dest[T_offset + static_cast<uint32_t>(idx)]) << std::endl;
                return detail::AssignmentDestination<T_Elem, T_Elem, memoryLayouts::contigous>{m_dest[T_offset + static_cast<uint32_t>(idx)]};
            }

            //returns ref to allow assignment
            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx)
            {
                static_assert(!std::is_same_v<bool, T_Elem>);

                const auto offset = laneCount_v<T_Elem>*static_cast<uint32_t>(idx);
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): &ctxVar=" << reinterpret_cast<uint64_t>(&m_dest) << std::endl;
                //std::cout << "WriteLeafXpr<dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](SimdLookupIndex): &ctxVar[0]=" << reinterpret_cast<uint64_t>(&m_dest[0]) << ", &ctxVar[offset=" << offset << "]=" << reinterpret_cast<uint64_t>(&m_dest[offset]) << std::endl;
                //printf("WriteLeafXpr<ctxVar> thread %d in block %d: before load\n", threadIdx.x, blockIdx.x);
                //auto tmp = m_dest[offset];
                //printf("WriteLeafXpr<ctxVar> thread %d in block %d: after  load, value is %d\n", threadIdx.x, blockIdx.x, tmp);

                return detail::AssignmentDestination<T_Elem, Pack_t<T_Elem, T_Elem>, memoryLayouts::contigous>{m_dest[laneCount_v<T_Elem>*static_cast<uint32_t>(idx)]};
            }

            XPR_ASSIGN_OPERATOR

        };

        //cannot be assigned to
        template<typename T_Elem, typename T_Lambda>
        class ReadLeafXpr<T_Elem, dataLocationTags::gatherScatterFunctor<T_Lambda>>{
            MemAccessorFunctor<T_Lambda> const m_source;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr ReadLeafXpr(MemAccessorFunctor<T_Lambda> const source) : m_source(source)
            {
            }

            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx) const
            {
                const auto & worker = idx.m_forEach.getWorker();
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): accessing index " << T_offset + static_cast<uint32_t>(idx) << std::endl;
                //std::cout << "ReadLeafXpr <dataLocationTags::ctxVar>(address=" << reinterpret_cast<uint64_t>(this) << ")::operator[](ScalarLookupIndex): value is: " << m_source[T_offset + static_cast<uint32_t>(idx)] << std::endl;
                return OneElemPack_t<T_Elem, T_Elem>{m_source[T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)]};
            }

            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx) const
            {
                static_assert(!std::is_same_v<bool, T_Elem>);

                //const auto & worker = idx.m_forEach.getWorker();
                //std::cout << "ReadLeafXpr <dataLocationTags::gatherScatterFunctor>::operator[](SimdLookupIndex): Worker " << worker.getWorkerIdx() << " accessing index " << static_cast<uint32_t>(idx) << ", simd-offset index=" << (laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))) << std::endl;

                return Pack_t<T_Elem, T_Elem>(MemAccessorFunctor([&idx, this](auto const i)constexpr{
                    const auto & worker = idx.m_forEach.getWorker();

                    //if(worker.getWorkerIdx()==0){std::cout << "Pack_t::CTOR: Worker " << worker.getWorkerIdx() << " accessing index " << i << ", simd-offset index=" << (i + laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))) << std::endl;}

                    return m_source[i + laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))];
                }));
            }
        };

        //can be assigned to
        template<typename T_Elem, typename T_Lambda>
        class WriteLeafXpr<T_Elem, dataLocationTags::gatherScatterFunctor<T_Lambda>>{
            MemAccessorFunctor<T_Lambda> const m_dest;

            static_assert(!std::is_const_v<T_Elem>);
            static_assert(!std::is_reference_v<T_Elem>);

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr WriteLeafXpr(MemAccessorFunctor<T_Lambda> const store) : m_dest(store)
            {
            }

            //returns ref to allow assignment
            template<typename T_Foreach, uint32_t T_offset>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](ScalarLookupIndex<T_Foreach, T_offset> const idx)
            {
                const auto & worker = idx.m_forEach.getWorker();
                return detail::AssignmentDestination<T_Elem, T_Elem, memoryLayouts::nonContigous<T_Lambda>>{m_dest, T_offset + worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx)};
            }

            //returns ref to allow assignment
            template<typename T_Foreach>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](SimdLookupIndex<T_Foreach> const idx)
            {
                static_assert(!std::is_same_v<bool, T_Elem>);
                const auto & worker = idx.m_forEach.getWorker();

                const auto offset = laneCount_v<T_Elem>*static_cast<uint32_t>(idx);

                return detail::AssignmentDestination<T_Elem, Pack_t<T_Elem, T_Elem>, memoryLayouts::nonContigous<T_Lambda>>{m_dest, laneCount_v<T_Elem> * (worker.getWorkerIdx() + worker.getNumWorkers() * static_cast<uint32_t>(idx))};
            }

            XPR_ASSIGN_OPERATOR

        };

        template<typename T_Functor, typename T_Operand>
        class UnaryXpr{

            static_assert(isXpr_v<T_Operand>);
            static_assert(!std::is_reference_v<T_Operand>);

            T_Operand m_operand;

        public:

            constexpr static bool allChildrenAreConst = allChildrenAreConst_v<UnaryXpr>;//std::is_const_v<T_Operand>;
            using ConstInfluencedAlias_t = std::conditional_t<allChildrenAreConst, const UnaryXpr, UnaryXpr>;

            constexpr UnaryXpr(UnaryXpr const&) = default;
            constexpr UnaryXpr(UnaryXpr &)      = default;
            constexpr UnaryXpr(UnaryXpr &&)     = default;

            template<typename T_OperandXpr>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr UnaryXpr(T_OperandXpr&& operand):m_operand(std::forward<T_OperandXpr>(operand))
            {
                static_assert(std::is_same_v<std::decay_t<T_OperandXpr>, std::decay_t<T_Operand>>);
            }

            //returns a unary Xpr that is const if both its operands were
            template<typename... T_Args>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr ConstInfluencedAlias_t makeConstIfPossible(T_Args&&... args){
                return UnaryXpr{std::forward<T_Args>(args)...};
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i)
            {
                static_assert(isPackWrapper_v<std::decay_t<decltype(T_Functor::SIMD_EVAL_F(m_operand[i]))>>);
                return T_Functor::SIMD_EVAL_F(m_operand[i]);
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                static_assert(isPackWrapper_v<std::decay_t<decltype(T_Functor::SIMD_EVAL_F(m_operand[i]))>>);
                return T_Functor::SIMD_EVAL_F(m_operand[i]);
            }
        };

        //Two operand expression objects combined via a Functor
        template<typename T_Functor, typename T_Left, typename T_Right>
        class BinaryXpr{

            static_assert(isXpr_v<T_Left> && isXpr_v<T_Right>);
            static_assert(!std::is_reference_v<T_Left> && !std::is_reference_v<T_Right>);

            T_Left m_leftOperand;
            T_Right m_rightOperand;

        public:

            constexpr static bool allChildrenAreConst = allChildrenAreConst_v<BinaryXpr>;
            using ConstInfluencedAlias_t = std::conditional_t<allChildrenAreConst, const BinaryXpr, BinaryXpr>;

            template<typename T_LeftXpr, typename T_RightXpr>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr BinaryXpr(T_LeftXpr&& left, T_RightXpr&& right):m_leftOperand(std::forward<T_LeftXpr>(left)), m_rightOperand(std::forward<T_RightXpr>(right))
            {
                static_assert(std::is_same_v<std::decay_t<T_Left>, std::decay_t<T_LeftXpr>>);
                static_assert(std::is_same_v<std::decay_t<T_Right>, std::decay_t<T_RightXpr>>);
            }

            constexpr BinaryXpr(BinaryXpr const&) = default;
            constexpr BinaryXpr(BinaryXpr &)      = default;
            constexpr BinaryXpr(BinaryXpr &&)     = default;

            //returns a binary Xpr that is const if both its operands were
            template<typename... T_Args>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr static ConstInfluencedAlias_t makeConstIfPossible(T_Args&&... args){
                return BinaryXpr{std::forward<T_Args>(args)...};
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i)
            {
                return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
            }

            XPR_ASSIGN_OPERATOR
        };

        template<typename T_Lambda, typename... T_Args>
        class NAryXpr{

            T_Lambda const m_lambda;
            std::tuple<T_Args...> m_args;

            static_assert(std::conjunction_v<trait::IsXpr<std::decay_t<T_Args>>...>);
            static_assert(!std::disjunction_v<std::is_reference<T_Args>...>);

            template<typename T_Idx, std::size_t... I>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) process(T_Idx const& idx, std::index_sequence<I...>) const {
                return m_lambda(std::get<I>(m_args)[idx]...);
            }

        public:

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr NAryXpr(T_Lambda const lambda, T_Args const... args):m_lambda(lambda), m_args(args...){}

            constexpr NAryXpr(NAryXpr const&) = default;
            constexpr NAryXpr(NAryXpr &)      = default;
            constexpr NAryXpr(NAryXpr &&)     = default;

            constexpr static bool allChildrenAreConst = allChildrenAreConst_v<NAryXpr>;
            using ConstInfluencedAlias_t = std::conditional_t<allChildrenAreConst, const NAryXpr, NAryXpr>;

            //returns an n-ary Xpr that is const if both its operands were
            template<typename... T_Args_>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr static ConstInfluencedAlias_t makeConstIfPossible(T_Args_&&... args){
                return NAryXpr{std::forward<T_Args_>(args)...};
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i)
            {
                return process(i, std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(m_args)> > >());
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
            {
                return process(i, std::make_index_sequence<std::tuple_size_v<std::decay_t<decltype(m_args)> > >());
            }
        };

        template<typename T_Foreach, typename T_Elem, typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void evaluateExpression(T_Foreach const& forEach, T_Xpr&& xpr)
        {
            constexpr auto lanes = laneCount_v<T_Elem>;
            constexpr auto numWorkers = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
            constexpr auto domainSize = std::decay_t<decltype(forEach)>::domainSize;

            constexpr auto simdLoops = domainSize/(numWorkers*lanes);
            constexpr auto elementsProcessedBySimd = simdLoops*numWorkers*lanes;
            constexpr auto leftovers = domainSize - elementsProcessedBySimd;

            //each worker has atleast this many leftover elems
            constexpr auto leftoversForAllThreads = (leftovers/numWorkers);
            constexpr auto leftoversProcessedByAllWorkers = leftoversForAllThreads * numWorkers;
            constexpr auto leftoversAfterEqualDist = leftovers - leftoversProcessedByAllWorkers;

            const auto workerIdx = forEach.getWorker().getWorkerIdx();

#if 0
            if(workerIdx == 0){
              const auto idOfThreadInGrid = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(forEach.getWorker().getAcc())[0u];
              std::cout << "\nevaluateExpression: workerIdx=" << workerIdx << ", threadInGrid=" << idOfThreadInGrid << " : running " << simdLoops << " simdLoops and " << leftoversForAllThreads << " scalar loops. numWorkers=" << numWorkers << ", domainSize=" << domainSize << ", laneCount=" << lanes << ", performCleanup=" << ((leftoversAfterEqualDist==0)?"false":"true") << std::endl;
            }
#endif

            for(uint32_t i = 0u; i<simdLoops; ++i){
                //uses the operator[] that returns const Pack_t
                //std::cout << "evaluateExpression: Worker " << workerIdx << ": beginning SimdLoop No. " << i << std::endl;
                std::forward<T_Xpr>(xpr)[SimdLookupIndex{forEach, i}];
                //std::cout << "evaluateExpression: Worker " << workerIdx << ": completed SimdLoop No. " << i << std::endl;
            }

            for(uint32_t i = 0u; i<leftoversForAllThreads; ++i){
                //uses the operator[] that returns const T_Elem &
                std::forward<T_Xpr>(xpr)[ScalarLookupIndex<T_Foreach, elementsProcessedBySimd>{forEach, i}];
            }

            if(workerIdx < leftoversAfterEqualDist){
                std::forward<T_Xpr>(xpr)[ScalarLookupIndex<T_Foreach, elementsProcessedBySimd+leftoversProcessedByAllWorkers>{forEach, 0u}];
            }
        }

        template<typename T_Foreach, typename T_Xpr>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto evalToCtxVar(T_Foreach const& forEach, T_Xpr&& xpr){
            //Xpr -> take 1st entry -> get type -> get Elem_t -> remove const
            using Elem_t = std::decay_t<typename std::decay_t<decltype(std::forward<T_Xpr>(xpr)[std::declval<ScalarLookupIndex<T_Foreach, 0u>>()])>::Elem_t>;
            using Config_t = typename std::decay_t<decltype(forEach)>::BaseConfig;
            using ContextVar_t = Variable<Elem_t, Config_t>;

            //make sure that the layout of the ctx var we generate is compatible with the Simd packs we use
            static_assert((Config_t::simdSize % laneCount_v<Elem_t>) == 0);

            ContextVar_t tmp;
            decltype(auto) storeXpr = (store(tmp) = std::forward<T_Xpr>(xpr));
            evaluateExpression<T_Foreach, Elem_t>(forEach, storeXpr);
            return tmp;
        }

        //single element, broadcasted if required
        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<std::decay_t<T_Elem>>, int> >
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(T_Elem&& elem){
            return ReadLeafXpr<std::decay_t<T_Elem>, dataLocationTags::scalar>(std::forward<T_Elem>(elem));
        }

        //pointer to threadblocks data
        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(T_Elem const * const ptr){
            return ReadLeafXpr<T_Elem, dataLocationTags::perBlockArray>(ptr);
        }

        //lockstep ctxVar
        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(lockstep::Variable<T_Elem, T_Config> const& ctxVar){
            return ReadLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>(ctxVar);
        }

        //load form non-contiguous memory via lambda
        template<typename T_Lambda>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr const auto load(MemAccessorFunctor<T_Lambda> func){
            using Elem_t = decltype(func[0]);
            return ReadLeafXpr<Elem_t, dataLocationTags::gatherScatterFunctor<T_Lambda>>(func);
        }

        template<typename T_Elem, std::enable_if_t<std::is_arithmetic_v<T_Elem>, int> >
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem & elem){
            return WriteLeafXpr<T_Elem, dataLocationTags::scalar>(elem);
        }

        template<typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Elem * const ptr){
            return WriteLeafXpr<T_Elem, dataLocationTags::perBlockArray>(ptr);
        }

        template<typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(lockstep::Variable<T_Elem, T_Config> & ctxVar){
            static_assert(!std::is_const_v<T_Elem>);
            return WriteLeafXpr<T_Elem, dataLocationTags::ctxVar<T_Config>>(ctxVar);
        }

        template<typename T_Lambda>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(MemAccessorFunctor<T_Lambda> func){
            using Elem_t = decltype(func[0]);
            //if passed lambda does not return references, we cannot assign
            static_assert(std::is_reference_v<Elem_t>);
            return WriteLeafXpr<std::remove_reference_t<Elem_t>, dataLocationTags::gatherScatterFunctor<T_Lambda>>(func);
        }

        template<typename T_Lambda, typename... T_Args>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto makeNAryNode(T_Lambda const func, T_Args... args){
            return NAryXpr<T_Lambda, T_Args...>(func, args...);
        }

//clean up
#undef XPR_OP_WRAPPER
#undef XPR_ASSIGN_OPERATOR

    } // namespace expr
} // namespace alpaka::lockstep

