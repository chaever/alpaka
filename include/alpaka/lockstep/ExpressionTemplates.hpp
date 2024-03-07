///TODO Copyright, License and other stuff here

#pragma once

//#include "Traits.hpp"

namespace alpaka::lockstep
{

    //prior to C++ 20 a static operator() is not allowed
#if __cplusplus < 202002L
    #define SIMD_EVAL_F eval
#else
    #define SIMD_EVAL_F operator()
#endif
/*
    //forward definition
    template<typename T_Functor, typename... T_Operands>
    class Xpr;

    //free function for Expression Evaluation
    ///TODO probably want a specialization for ASTs wherein the root is an assignment (T_Xpr has form Xpr<Assign,T_Left,T_Right>), as we then want to return void/Vec<elem>& instead of Vec<elem&>?
    template<typename T_Xpr>
    auto evalXpr(const T_Xpr& xpr){




        //if we lookup by uint32_t, we get a single element of the result expression
        using OneElement_t = decltype(std::declval<T_Xpr>().getValueAtIndex(std::declval<uint32_t>()));


        using ResultVectorWith_t = trait::Xpr_t_to_Vector_Container_t<>;

        using VectorWithNewElem_t = ResultVectorWithoutElem_t<OneElement_t>;

        auto VectorWithNewElem_t tmp;

        constexpr auto lanes = laneCount<OneElement_t>;
        constexpr auto sizeOfResultType = ???;///TODO need trait that specializes for Xpr&Leaf nodes (Arrays etc) -> TraitName_v<VectorWithNewElem_t>
        constexpr auto vectorLoops = sizeOfResultType/lanes;

        //get pointer to start of internal data storage
        T_Type* ptr = &tmp[0];

        for(std::size_t i = 0u; i<vectorLoops; ++i, ptr+=lanes){
            //uses the getValueAtIndex that returns Pack_t
            SimdInterface_t<T_Type>::storeUnaligned(xpr.getValueAtIndex(SimdLookupIndex<T_Type>(i)), ptr);
        }
        for(std::size_t i = vectorLoops*lanes; i<sizeOfResultType; ++i, ++ptr){
            //uses the getValueAtIndex that returns T_Type
            *ptr = xpr.getValueAtIndex(i);
        }
        return tmp;
    }

    //void-terminated AST, need an assignment to have effect inbetween
    void evalXpr(const T_Xpr& xpr){
        static constexpr auto sizeOfResult T_Xpr::size;

        constexpr auto lanes = laneCount<OneElement_t>;
        constexpr auto sizeOfResultType = ???;///TODO need trait that specializes for Xpr&Leaf nodes (Arrays etc) -> TraitName_v<VectorWithNewElem_t>
        constexpr auto vectorLoops = sizeOfResultType/lanes;

        //get pointer to start of internal data storage
        T_Type* ptr = &tmp[0];

        for(std::size_t i = 0u; i<vectorLoops; ++i, ptr+=lanes){
            //uses the getValueAtIndex that returns Pack_t
            SimdInterface_t<T_Type>::storeUnaligned(xpr.getValueAtIndex(SimdLookupIndex<T_Type>(i)), ptr);
        }
        for(std::size_t i = vectorLoops*lanes; i<sizeOfResultType; ++i, ++ptr){
            //uses the getValueAtIndex that returns T_Type
            *ptr = xpr.getValueAtIndex(i);
        }

    }*/

    struct Identity{
        template<typename T_Type>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Type t){
            return t;
        }
    };

    struct Addition{
        //should also work for Simd-types that define their own operator=
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left left, T_Right right){
            //uses T_Left::operator+(T_Right)
            return left+right;
        }
    };

    struct Assignment{
        //should also work for Simd-types that define their own operator+
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left left, T_Right right){
            //uses T_Left::operator=(T_Right)
            return left=right;
        }
    };

    template<typename T_Functor, typename... T_Operands>
    class Xpr{
        T_Operands&... m_operands;///TODO when instanciating Xpr<Assign, TA, TB> the TA operand needs to be a non-const & as we assign to it
    public:

        Xpr(T_Operands const&... operands) : m_operands(operands)
        {
        }

        template<typename T_Idx>
        constexpr auto const getValueAtIndex(T_Idx const i) const
        {
            return T_Functor::SIMD_EVAL_F(m_operands.getValueAtIndex(i)...);
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const {
            using ThisXpr_t = Xpr<T_Functor, T_Operands...>;
            return Xpr<Addition, ThisXpr_t, T_Other>(*this, other);
        }

        template<typename T_Other>
        constexpr auto operator=(T_Other const & other) const {
            using ThisXpr_t = Xpr<T_Functor, T_Operands...>;
            return Xpr<Assignment, ThisXpr_t, T_Other>(*this, other);
        }
/*
        template<typename... T_Args>
        auto eval(T_Args... args) const {
            return alpaka::lockstep::evalXpr(*this, std::forward(args)...);
        }*/
    };

    ///TODO could be reused to broadcast scalar values (3->vector<>(3))
    template<typename T_Type>
    makeXpr(T_Type const& t){
        //wrap whatever we got in a unary Xpr that just returns the types values
        return Xpr<Identity, T_Type>(t);
    }

} // namespace alpaka::lockstep
