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
    template<bool T_assumeOneWorker, typename T_Foreach, typename T>
    class ReadLeafXpr;
    template<bool T_assumeOneWorker, typename T_Foreach, typename T>
    class WriteLeafXpr;

    template<typename T_Type, typename T_Config>
    struct Variable;

    template<typename T_Worker, typename T_Config>
    class ForEach;

    template<typename T_ForEach, typename T_Elem>
    auto load(T_ForEach const& forEach, T_Elem const * const ptr);

    template<typename T_Worker, typename T_Elem, typename T_Config>
    auto load(lockstep::ForEach<T_Worker, T_Config> const& forEach, typename lockstep::Variable<T_Elem, T_Config> const& ctxVar);

    template<typename T_ForEach, typename T_Elem>
    auto store(T_ForEach const& forEach, T_Elem * const ptr);

    template<typename T_Worker, typename T_Elem, typename T_Config>
    auto store(lockstep::ForEach<T_Worker, T_Config> const& forEach, typename lockstep::Variable<T_Elem, T_Config> & ctxVar);

    namespace detail
    {
        template<typename T>
        struct IsXpr{
            static constexpr bool value = false;
        };

        template<typename T_Functor, typename T_Left, typename T_Right, typename T_Foreach>
        struct IsXpr<Xpr<T_Functor, T_Left, T_Right, T_Foreach>>{
            static constexpr bool value = true;
        };

        template<bool T_assumeOneWorker, typename T_Foreach, typename T>
        struct IsXpr<ReadLeafXpr<T_assumeOneWorker, T, T_Foreach> >{
            static constexpr bool value = true;
        };

        template<bool T_assumeOneWorker, typename T_Foreach, typename T>
        struct IsXpr<WriteLeafXpr<T_assumeOneWorker, T, T_Foreach> >{
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
            //additions right operand is read-only
            return load(forEach, other);
        }

        template<typename T_Other, typename T_Foreach, std::enable_if_t< detail::IsXpr<T_Other>::value, int> = 0>
        static decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){
            return other;
        }
    };

    struct Assignment{
        template<typename T_Left, typename T_Right, std::enable_if_t< std::is_same_v<T_Right, Pack_t<T_Left>>, int> = 0>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const right){
            //std::cout << "Assignment<Pack>::operator[]: before writing " << right[0] << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;
            SimdInterface_t<T_Left>::storeUnaligned(right, &left);
            //std::cout << "Assignment<Pack>::operator[]: after  writing " << left     << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;
            return right;
        }
        template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_same_v<T_Right, Pack_t<T_Left>>, int> = 0>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
            //std::cout << "Assignment<Scalar>::operator[]: before writing " << right << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;
            return left=right;
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
            //assignmnets right operand is read-only
            return load(forEach, other);
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
    template<bool T_assumeOneWorker, typename T_Foreach, typename T>
    class ReadLeafXpr{
        T const& m_source;
    public:
        T_Foreach const& m_forEach;

        static constexpr bool assumeOneWorker = T_assumeOneWorker;

        //takes a ptr that points to start of domain
        ReadLeafXpr(T_Foreach const& forEach, T const * const source) : m_source(*source), m_forEach(forEach)
        {
        }

        //allows making an expression from CtxVariable
        template<typename T_Config>
        ReadLeafXpr(T_Foreach const& forEach, typename lockstep::Variable<T, T_Config> const& v) : m_source(v[0u]), m_forEach(forEach)
        {
        }

        decltype(auto) operator[](SimdLookupIndex const idx) const
        {
            //auto* tmpPtr = &m_source + laneCount<T_Elem> * m_forEach.getWorker().getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(m_forEach.getWorker())>::numWorkers) * static_cast<uint32_t>(idx);
            //std::cout << "ReadLeafXpr::operator[]<T_assumeOneWorker=" << (T_assumeOneWorker?"true":"false") << ", SimdLookupIndex>("<<static_cast<uint32_t>(idx)<<"): loading from " << reinterpret_cast<uint64_t>(tmpPtr) << " = " << reinterpret_cast<uint64_t>(&m_source) << "+" << (reinterpret_cast<uint64_t>(tmpPtr)-reinterpret_cast<uint64_t>(&m_source)) << std::endl;
            //const auto& tmp = SimdInterface_t<T_Elem>::loadUnaligned(tmpPtr);
            //std::cout << "ReadLeafXpr::operator[]("<<static_cast<uint32_t>(idx)<<")[0] = " << tmp[0] << std::endl;

            return SimdInterface_t<T>::loadUnaligned(&m_source + laneCount<T> * (m_forEach.getWorker().getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(m_forEach.getWorker())>::numWorkers) * static_cast<uint32_t>(idx)));
        }

        decltype(auto) operator[](ScalarLookupIndex const idx) const
        {
            //auto* tmpPtr = &m_source + m_forEach.getWorker().getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(m_forEach.getWorker())>::numWorkers) * static_cast<uint32_t>(idx);
            //std::cout << "ReadLeafXpr::operator[]<T_assumeOneWorker=" << (T_assumeOneWorker?"true":"false") << ", ScalarLookupIndex>("<<static_cast<uint32_t>(idx)<<"): loading from " << reinterpret_cast<uint64_t>(tmpPtr) << " = " << reinterpret_cast<uint64_t>(&m_source) << "+" << (reinterpret_cast<uint64_t>(tmpPtr)-reinterpret_cast<uint64_t>(&m_source)) << std::endl;
            //const auto& tmp = *tmpPtr;
            //std::cout << "ReadLeafXpr::operator[]("<<static_cast<uint32_t>(idx)<<") = " << tmp << std::endl;

            return (&m_source)[m_forEach.getWorker().getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(m_forEach.getWorker())>::numWorkers) * static_cast<uint32_t>(idx)];
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            using Op = Addition;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }
    };

    //can be assigned to
    //can be made from pointers, or some container classes
    template<bool T_assumeOneWorker, typename T_Foreach, typename T>
    class WriteLeafXpr{
        T & m_dest;
    public:
        T_Foreach const& m_forEach;

        static constexpr bool assumeOneWorker = T_assumeOneWorker;

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
        auto & operator[](ScalarLookupIndex const idx) const
        {
            auto const& worker = m_forEach.getWorker();
            return (&m_dest)[m_forEach.getWorker().getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(m_forEach.getWorker())>::numWorkers) * static_cast<uint32_t>(idx)];
        }

        //returns ref to allow assignment
        auto & operator[](SimdLookupIndex const idx) const
        {
            auto const& worker = m_forEach.getWorker();
            return (&m_dest)[laneCount<T> * (worker.getWorkerIdx() + (T_assumeOneWorker ? 1 : std::decay_t<decltype(worker)>::numWorkers) * static_cast<uint32_t>(idx))];
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr decltype(auto) operator=(T_Other const & other) const
        {
            using Op = Assignment;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach>(*this, rightXpr);
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

        Xpr(T_Left_const_t left, T_Right_const_t right):m_leftOperand(left), m_rightOperand(right), m_forEach(left.m_forEach)
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
            return Xpr<Op, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }

        //used if T_Other is already an expression
        template<typename T_Other>
        constexpr decltype(auto) operator=(T_Other const & other) const
        {
            using Op = Assignment;
            auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);
            return Xpr<Op, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach>(*this, rightXpr);
        }
    };

    ///TODO T_Elem should maybe also be deduced?
    template<typename T_Elem, typename T_Xpr>
    void evaluateExpression(T_Xpr const& xpr)
    {
        constexpr auto lanes = laneCount<T_Elem>;
        constexpr auto numWorkers = std::decay_t<decltype(xpr.m_forEach.getWorker())>::numWorkers;
        constexpr auto domainSize = std::decay_t<decltype(xpr.m_forEach)>::domainSize;

        constexpr auto simdLoops = domainSize/(numWorkers*lanes);

        constexpr auto lengthOfVectors = alpaka::core::divCeil(domainSize, numWorkers);
        constexpr auto vectorLoops = lengthOfVectors/(lanes*numWorkers);
        const auto workerIdx = xpr.m_forEach.getWorker().getWorkerIdx();

        //std::cout << "evaluateExpression: running " << vectorLoops << " vectorLoops and " << (lengthOfVectors - vectorLoops*lanes) << " scalar loops." << std::endl;

        for(std::size_t i = 0u; i<vectorLoops; ++i){
            //std::cout << "evaluateExpression: starting vectorLoop " << i << std::endl;
            //uses the operator[] that returns const Pack_t
            xpr[SimdLookupIndex(i)];

            //std::cout << "evaluateExpression: finished vectorLoop " << i << std::endl;
        }
        for(std::size_t i = lanes*vectorLoops; i<lengthOfVectors; ++i){
            //std::cout << "evaluateExpression: starting scalarLoop " << i << std::endl;
            //uses the operator[] that returns const T_Elem &
            xpr[ScalarLookupIndex(i)];
        }
    }

    template<typename T_ForEach, typename T_Elem>
    auto load(T_ForEach const& forEach, T_Elem const * const ptr){
        static constexpr auto assumeOnlyOneWorkerWillWorkOnTheData = false;
        return ReadLeafXpr<assumeOnlyOneWorkerWillWorkOnTheData, T_ForEach, T_Elem>(forEach, ptr);
    }

    template<typename T_Worker, typename T_Elem, typename T_Config>
    auto load(lockstep::ForEach<T_Worker, T_Config> const& forEach, typename lockstep::Variable<T_Elem, T_Config> const& ctxVar){
        using ForEach_t = ForEach<T_Worker, T_Config>;
        static constexpr auto assumeOnlyOneWorkerWillWorkOnTheData = true;
        return ReadLeafXpr<assumeOnlyOneWorkerWillWorkOnTheData, ForEach_t, T_Elem>(forEach, ctxVar);
    }

    template<typename T_ForEach, typename T_Elem>
    auto store(T_ForEach const& forEach, T_Elem * const ptr){
        static constexpr auto assumeOnlyOneWorkerWillWorkOnTheData = false;
        return WriteLeafXpr<assumeOnlyOneWorkerWillWorkOnTheData, T_ForEach, T_Elem>(forEach, ptr);
    }

    template<typename T_Worker, typename T_Elem, typename T_Config>
    auto store(lockstep::ForEach<T_Worker, T_Config> const& forEach, typename lockstep::Variable<T_Elem, T_Config> & ctxVar){
        using ForEach_t = ForEach<T_Worker, T_Config>;
        static constexpr auto assumeOnlyOneWorkerWillWorkOnTheData = true;
        return WriteLeafXpr<assumeOnlyOneWorkerWillWorkOnTheData, ForEach_t, T_Elem>(forEach, ctxVar);
    }
    ///TODO need function that returns CtxVar
    //void evalToCtxVar

} // namespace alpaka::lockstep
