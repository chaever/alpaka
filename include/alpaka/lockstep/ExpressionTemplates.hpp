///TODO Copyright, License and other stuff here

#pragma once

#include "Simd.hpp"

namespace alpaka::lockstep
{

    //prior to C++ 20 a static operator() is not allowed
#if __cplusplus < 202002L
    #define SIMD_EVAL_F eval
#else
    #define SIMD_EVAL_F operator()
#endif

    //forward declarations
    template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach>
    class Xpr;
    template<typename T_Foreach, typename T>
    class ReadLeafXpr;
    template<typename T_Foreach, typename T>
    class WriteLeafXpr;

    template<typename T_Type, typename T_Config>
    struct Variable;

    namespace detail
    {
        template<typename T_Idx>
        struct IndexOperatorLeafRead;

        template<typename T_Idx>
        struct IndexOperatorLeafRead{

            template<typename T_Elem>
            static T_Elem const& eval(T_Idx idx, T_Elem const * const ptr)
            {
                //std::cout << "IndexOperatorLeafRead<uint32_t>::eval("<<static_cast<uint32_t>(idx)<<"): loading from " << reinterpret_cast<uint64_t>(ptr + static_cast<uint32_t>(idx)) << " , base is " << reinterpret_cast<uint64_t>(ptr) << std::endl;

                return ptr[idx];
            }
        };

        //specialization for SIMD-XprLookupIndex
        //returns only const Packs because they are copies
        template<typename T_Elem>
        struct IndexOperatorLeafRead<SimdLookupIndex<T_Elem>>{

            static typename SimdLookupIndex<T_Elem>::Pack_type const eval(SimdLookupIndex<T_Elem> idx, T_Elem const * const ptr)
            {
                //std::cout << "IndexOperatorLeafRead<XprLookupIndex>::eval("<<static_cast<uint32_t>(idx)<<"): loading from " << reinterpret_cast<uint64_t>(ptr + static_cast<uint32_t>(idx)) << " , base is " << reinterpret_cast<uint64_t>(ptr) << std::endl;
                //const auto& tmp = SimdInterface_t<T_Type>::loadUnaligned(ptr + static_cast<uint32_t>(idx));
                //std::cout << "IndexOperatorLeafRead<XprLookupIndex>::eval("<<static_cast<uint32_t>(idx)<<") = " << tmp << std::endl;

                return SimdInterface_t<T_Elem>::loadUnaligned(ptr + static_cast<uint32_t>(idx));
            }
        };

        template<typename T_Elem>
        struct IndexOperatorLeafRead<ScalarLookupIndex<T_Elem>>{

            static T_Elem const& eval(ScalarLookupIndex<T_Elem> idx, T_Elem const * const ptr)
            {
                //std::cout << "IndexOperatorLeafRead<uint32_t>::eval("<<static_cast<uint32_t>(idx)<<"): loading from " << reinterpret_cast<uint64_t>(ptr + static_cast<uint32_t>(idx)) << " , base is " << reinterpret_cast<uint64_t>(ptr) << std::endl;

                return ptr[static_cast<uint32_t>(idx)];
            }
        };

        template<typename T_Left, typename T_Right>
        struct AssignmentTrait{
            //assign scalars
            static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
                //uses T_Left::operator=(T_Right)
                return left=right;
            }
        };

        template<typename T_Left>
        struct AssignmentTrait<T_Left, Pack_t<T_Left>>{
            //assign packs
            static constexpr const auto SIMD_EVAL_F(T_Left& left, Pack_t<T_Left> const& right){
                SimdInterface_t<T_Left>::storeUnaligned(right, &left);
                return right;//return right to preserve semantics of operator=
            }
        };

        template<typename T>
        struct IsXpr{
            static constexpr bool value = false;
        };

        template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach>
        struct IsXpr<Xpr<T_Functor, T_Left, T_Right, T_Foreach>>{
            static constexpr bool value = true;
        };

        template<typename T, typename T_Foreach>
        struct IsXpr<ReadLeafXpr<T, T_Foreach> >{
            static constexpr bool value = true;
        };

        template<typename T, typename T_Foreach>
        struct IsXpr<WriteLeafXpr<T, T_Foreach> >{
            static constexpr bool value = true;
        };

    } // namespace detail

    struct Addition{
        //should also work for Simd-types that define their own operator+
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){
            //uses T_Left::operator+(T_Right)
            return left+right;
        }

        template<typename T>
        struct OperandXprTrait{
            //left & right parent expressions shall both be const
            using LeftArg_t = T const;
            using RightArg_t = T const;
        };

        //converts the righthand container operand to an expression if that is needed
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!detail::IsXpr<T_Other>::value, int> = 0>
        static decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
            //additions right operand is only read
            return ReadLeafXpr(forEach, other);
        }

        template<typename T_Other, typename T_Foreach, std::enable_if_t< detail::IsXpr<T_Other>::value, int> = 0>
        static decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
            return other;
        }
    };

    struct Assignment{
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
            return detail::AssignmentTrait<T_Left, T_Right>::SIMD_EVAL_F(left, right);
        }

        template<typename T>
        struct OperandXprTrait{
            //right parent expression is const, left is assignee and therefore not const
            using LeftArg_t = T;
            using RightArg_t = T const;
        };

        //converts the righthand container operand to an expression if that is needed
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!detail::IsXpr<T_Other>::value, int> = 0>
        static decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
            //additions right operand is only read
            return ReadLeafXpr(forEach, other);
        }

        template<typename T_Other, typename T_Foreach, std::enable_if_t< detail::IsXpr<T_Other>::value, int> = 0>
        static decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
            return other;
        }
    };

    //shortcuts
    template<typename T_Functor, typename T>
    using XprArgLeft_t = typename T_Functor::template OperandXprTrait<T>::LeftArg_t;
    template<typename T_Functor, typename T>
    using XprArgRight_t = typename T_Functor::template OperandXprTrait<T>::LeftArg_t;


    //cannot be assigned to
    //can be made from pointers, or some container classes
    template<typename T_Foreach, typename T>
    class ReadLeafXpr{
        T const& m_source;
    public:
        T_Foreach const& m_forEach;

        using ThisXpr_t = ReadLeafXpr<T_Foreach, T>;
        static constexpr bool simdWidthIsUnspecified = T_Foreach::simdWidthIsUnspecified;

        //takes a ptr that points to start of domain
        ReadLeafXpr(T_Foreach const& forEach, T const * const source) : m_source(*source), m_forEach(forEach)
        {
        }

        //allows making an expression from CtxVariable
        template<typename T_Config>
        ReadLeafXpr(T_Foreach const& forEach, typename lockstep::Variable<T, T_Config> const& v) : m_source(v[0u]), m_forEach(forEach)
        {
        }

        //returns const ref, since Leaves are not meant to be assigned to
        template<typename T_Idx>
        decltype(auto) operator[](T_Idx const idx) const
        {
            //std::cout << "ReadLeafXpr::operator[]: evaluating IndexOperatorLeafRead::eval(" << static_cast<uint32_t>(idx) << ")..." << std::endl;
            //const auto& tmp = detail::IndexOperatorLeafRead<T_Idx>::eval(idx, &m_source);
            //std::cout << "ReadLeafXpr::operator[]: IndexOperatorLeafRead::eval(" << static_cast<uint32_t>(idx) << ") = " << tmp << std::endl;

            return detail::IndexOperatorLeafRead<T_Idx>::eval(idx, &m_source);
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            using Op = Addition;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, ThisXpr_t, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }
    };

    //can be assigned to
    //can be made from pointers, or some container classes
    template<typename T_Foreach, typename T>
    class WriteLeafXpr{
        T & m_dest;
    public:
        T_Foreach const& m_forEach;

        using ThisXpr_t = WriteLeafXpr<T_Foreach, T>;
        static constexpr bool simdWidthIsUnspecified = T_Foreach::simdWidthIsUnspecified;

        //takes a ptr that points to start of domain
        WriteLeafXpr(T_Foreach const& forEach, T * const dest) : m_dest(*dest), m_forEach(forEach)
        {
        }

        //allows making an expression from CtxVariable
        template<typename T_Config>
        WriteLeafXpr(T_Foreach const& forEach, typename lockstep::Variable<T, T_Config> & v) : m_dest(v[0u]), m_forEach(forEach)
        {
        }

        //returns ref to allow assignment
        template<typename T_Idx>
        auto & operator[](T_Idx const idx) const
        {
            //static_cast will turn XprLookupIndex into a flattened value when required
            return (&m_dest)[static_cast<uint32_t>(idx)];
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr decltype(auto) operator=(T_Other const & other) const
        {
            using Op = Assignment;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, ThisXpr_t, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }
    };

    //const left operand, cannot assign
    template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach>
    class Xpr{

        //left&right with constness added as required by the Functor
        using T_Left_const_t = XprArgLeft_t<T_Functor, T_Left>;
        using T_Right_const_t = XprArgRight_t<T_Functor, T_Right>;

        T_Left_const_t m_leftOperand;
        T_Right_const_t m_rightOperand;
    public:
        T_Foreach const& m_forEach;

        using ThisXpr_t = Xpr<T_Functor, T_Left, T_Right, T_Foreach>;
        static constexpr bool simdWidthIsUnspecified = T_Foreach::simdWidthIsUnspecified;

        Xpr(T_Left_const_t left, T_Right_const_t right):m_leftOperand(left), m_rightOperand(right), m_forEach(decltype(left)::simdWidthIsUnspecified ? right.m_forEach : left.m_forEach )
        {
        }

        template<typename T_Idx>
        constexpr decltype(auto) operator[](T_Idx const i) const
        {
            //std::cout << "Xpr::operator[]: evaluating m_leftOperand[" << static_cast<uint32_t>(i) << "]..." << std::endl;
            //const auto& tmp1 = m_leftOperand[i];
            //std::cout << "Xpr::operator[]: m_leftOperand[" << static_cast<uint32_t>(i) << "] = " << tmp1 << std::endl;
            //std::cout << "Xpr::operator[]: evaluating m_rightOperand[" << static_cast<uint32_t>(i) << "]..." << std::endl;
            //const auto& tmp2 = m_rightOperand[i];
            //std::cout << "Xpr::operator[]: m_rightOperand[" << static_cast<uint32_t>(i) << "] = " << tmp2 << std::endl;

            return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr decltype(auto) operator+(T_Other const & other) const
        {
            using Op = Addition;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, ThisXpr_t, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr decltype(auto) operator=(T_Other const & other) const
        {
            using Op = Assignment;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, ThisXpr_t, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }
    };

    ///TODO lengthOfVectors should be deduced (not automagically by the compiler, but by the user via traits)
    ///TODO T_Elem should maybe also be deduced?
    template<typename T_Elem, std::size_t lengthOfVectors, typename T_Xpr>
    void evaluateExpression(T_Xpr const& xpr)
    {
        constexpr auto lanes = laneCount<T_Elem>;
        constexpr auto vectorLoops = lengthOfVectors/lanes;

        //std::cout << "evaluateExpression: running " << vectorLoops << " vectorLoops and " << (lengthOfVectors - vectorLoops*lanes) << " scalar loops." << std::endl;

        for(std::size_t i = 0u; i<vectorLoops; ++i){
            //std::cout << "evaluateExpression: starting vectorLoop " << i << std::endl;
            //uses the operator[] that returns const Pack_t
            xpr[SimdLookupIndex<T_Elem>(i)];

            //std::cout << "evaluateExpression: finished vectorLoop " << i << std::endl;
        }
        for(std::size_t i = vectorLoops*lanes; i<lengthOfVectors; ++i){
            //uses the operator[] that returns const T_Elem &
            xpr[ScalarLookupIndex<T_Elem>(i)];
        }
    }

    ///TODO need function that returns CtxVar
    //void evalToCtxVar


} // namespace alpaka::lockstep
