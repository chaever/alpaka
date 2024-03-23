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
    template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
    class ReadLeafXpr;
    template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
    class WriteLeafXpr;

    template<typename T_Type, typename T_Config>
    struct Variable;

    template<typename T_ForEach, typename T_Elem>
    constexpr auto load(T_ForEach const& forEach, T_Elem const * const ptr);

    template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
    constexpr auto load(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> const& ctxVar);

    template<typename T_ForEach, typename T_Elem>
    constexpr auto store(T_ForEach const& forEach, T_Elem * const ptr);

    template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
    constexpr auto store(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> & ctxVar);

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

        template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
        struct IsXpr<ReadLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride> >{
            static constexpr bool value = true;
        };

        template<typename T_Foreach, typename T_Elem, uint32_t T_dimensions, uint32_t T_stride>
        struct IsXpr<WriteLeafXpr<T_Elem, T_Foreach, T_dimensions, T_stride> >{
            static constexpr bool value = true;
        };
    } // namespace detail

    template<typename T>
    static constexpr bool isXpr_v = detail::IsXpr<T>::value;

//operations like +,-,*,/ that dont modify their operands
#define BINARY_READONLY_OP(name, shortFunc)\
    struct name{\
        template<typename T_Left, typename T_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){\
            return left shortFunc right;\
        }\
        template<typename T>\
        struct OperandXprTrait{\
            using LeftArg_t = T const;\
            using RightArg_t = T const;\
        };\
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
            return load(forEach, other);\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
            return other;\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
            return load(forEach, other);\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
            return other;\
        }\
    };

//used for =, +=, *=,...
#define BINARY_ASSIGNMENT_OP(name, shortFunc)\
    struct name{\
        template<typename T_Left, typename T_Right, std::enable_if_t< std::is_same_v<T_Right, Pack_t<T_Left>>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const right){\
            /*std::cout << #name << "<Pack>::operator[]: before writing " << right[0] << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/\
            SimdInterface_t<T_Left>::storeUnaligned(right, &left);\
            /*std::cout << #name << "<Pack>::operator[]: after  writing " << left     << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/\
            return right;\
        }\
        template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_same_v<T_Right, Pack_t<T_Left>>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){\
            /*std::cout << #name << "<Scalar>::operator[]: before writing " << right << " to " << reinterpret_cast<uint64_t>(&left) << std::endl;*/\
            return left=right;\
        }\
        template<typename T>\
        struct OperandXprTrait{\
            using LeftArg_t = T;\
            using RightArg_t = T const;\
        };\
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
            return load(forEach, other);\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeRightXprFromContainer(T_Other const& other, T_Foreach const& forEach){\
            return other;\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t<!isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
            return store(forEach, other);\
        }\
        template<typename T_Other, typename T_Foreach, std::enable_if_t< isXpr_v<T_Other>, int> = 0>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr decltype(auto) makeLeftXprFromContainer(T_Other & other, T_Foreach const& forEach){\
            return other;\
        }\
    };

    BINARY_READONLY_OP(Addition, +)
    BINARY_READONLY_OP(Subtraction, -)
    BINARY_READONLY_OP(Multiplication, *)
    BINARY_READONLY_OP(Division, /)

    BINARY_ASSIGNMENT_OP(Assignment, =)
    BINARY_ASSIGNMENT_OP(AssignAdd, +=)
    BINARY_ASSIGNMENT_OP(AssignMul, *=)

//clean up
#undef BINARY_READONLY_OP
#undef BINARY_ASSIGNMENT_OP

//needed since there is no space between "operator" and "+" in "operator+"
#define XPR_OP_WRAPPER() operator

//for operator definitions inside the Xpr classes (=, +=, *= etc are not allowed as non-member functions).
//Expression must be the lefthand operand(this).
#define XPR_OP(name, shortFunc)\
    template<typename T_Other>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Other const & other) const\
    {\
        using Op = name;\
        auto rightXpr = Op::makeRightXprFromContainer(other, m_forEach);\
        return Xpr<Op, std::decay_t<decltype(*this)>, decltype(rightXpr), T_Foreach>(*this, rightXpr);\
    }

//free operator definitions. Expression can be the left or right operand.
#define XPR_OPERATOR(name, shortFunc)\
    template<typename T_Left, typename T_Right, std::enable_if_t< alpaka::lockstep::isXpr_v<T_Left>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left left, T_Right const& right)\
    {\
        auto rightXpr = alpaka::lockstep::name::makeRightXprFromContainer(right, left.m_forEach);\
        return alpaka::lockstep::Xpr<alpaka::lockstep::name, T_Left, decltype(rightXpr), decltype(left.m_forEach)>(left, rightXpr);\
    }\
    template<typename T_Left, typename T_Right, std::enable_if_t<!alpaka::lockstep::isXpr_v<T_Left> && alpaka::lockstep::isXpr_v<T_Right>, int> = 0>\
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto XPR_OP_WRAPPER()shortFunc(T_Left const& left, T_Right right)\
    {\
        auto leftXpr = alpaka::lockstep::name::makeLeftXprFromContainer(left, right.m_forEach);\
        return alpaka::lockstep::Xpr<alpaka::lockstep::name, decltype(leftXpr), T_Right, decltype(right.m_forEach)>(leftXpr, right);\
    }

    XPR_OPERATOR(Addition, +)
    XPR_OPERATOR(Subtraction, -)
    XPR_OPERATOR(Multiplication, *)
    XPR_OPERATOR(Division, /)

#undef XPR_OPERATOR

    //shortcuts
    template<typename T_Functor, typename T>
    using XprArgLeft_t = typename T_Functor::template OperandXprTrait<T>::LeftArg_t;
    template<typename T_Functor, typename T>
    using XprArgRight_t = typename T_Functor::template OperandXprTrait<T>::LeftArg_t;

    //scalar, read-only node
    template<typename T_Foreach, typename T_Elem, uint32_t T_stride>
    class ReadLeafXpr<T_Foreach, T_Elem, 0u, T_stride>{
        T_Elem const& m_source;
    public:
        T_Foreach const& m_forEach;

        ReadLeafXpr(T_Foreach const& forEach, T_Elem const& source) : m_source(source), m_forEach(forEach)
        {
        }

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] SimdLookupIndex const idx) const
        {
            return SimdInterface_t<T_Elem>::broadcast(m_source);
        }

        template<uint32_t T_offset>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_offset> const idx) const
        {
            return m_source;
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

        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] SimdLookupIndex const idx) const
        {
            return SimdInterface_t<T_Elem>::broadcast(m_dest);
        }

        template<uint32_t T_offset>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[]([[unused]] ScalarLookupIndex<T_offset> const idx) const
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
            return SimdInterface_t<T_Elem>::loadUnaligned(&m_source + laneCount<T_Elem> * (m_forEach.getWorker().getWorkerIdx() + T_stride * static_cast<uint32_t>(idx)));
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
            return (&m_dest)[laneCount<T_Elem> * (worker.getWorkerIdx() + T_stride * static_cast<uint32_t>(idx))];
        }

        XPR_OP(Assignment, =)
        XPR_OP(AssignAdd, +=)
        XPR_OP(AssignMul, *=)
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
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) operator[](T_Idx const i) const
        {
            return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
        }

        XPR_OP(Assignment, =)
        XPR_OP(AssignAdd, +=)
        XPR_OP(AssignMul, *=)
    };

    ///TODO T_Elem should maybe also be deduced?
    template<typename T_Elem, typename T_Xpr>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void evaluateExpression(T_Xpr const& xpr)
    {
        constexpr auto lanes = laneCount<T_Elem>;
        constexpr auto numWorkers = std::decay_t<decltype(xpr.m_forEach.getWorker())>::numWorkers;
        constexpr auto domainSize = std::decay_t<decltype(xpr.m_forEach)>::domainSize;

        constexpr auto simdLoops = domainSize/(numWorkers*lanes);

        constexpr auto elementsProcessedBySimd = simdLoops*lanes*numWorkers;

        const auto workerIdx = xpr.m_forEach.getWorker().getWorkerIdx();

        //std::cout << "evaluateExpression: running " << simdLoops << " simdLoops and " << (domainSize - simdLoops*lanes*numWorkers) << " scalar loops." << std::endl;

        for(std::size_t i = 0u; i<simdLoops; ++i){
            //std::cout << "evaluateExpression: starting vectorLoop " << i << std::endl;
            //uses the operator[] that returns const Pack_t
            xpr[SimdLookupIndex(i)];
            //std::cout << "evaluateExpression: finished vectorLoop " << i << std::endl;
        }
        for(std::size_t i = 0u; i<(domainSize-elementsProcessedBySimd); ++i){
            //std::cout << "evaluateExpression: starting scalarLoop " << i << std::endl;
            //uses the operator[] that returns const T_Elem &
            xpr[ScalarLookupIndex<elementsProcessedBySimd>(i)];
        }
    }

    //single element, broadcasted if required
    template<typename T_ForEach, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_ForEach const& forEach, T_Elem const & elem){
        return ReadLeafXpr<T_ForEach, T_Elem, 0u, 0u>(forEach, elem);
    }

    //pointer to threadblocks data
    template<typename T_ForEach, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_ForEach const& forEach, T_Elem const * const ptr){
        constexpr uint32_t stride = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
        return ReadLeafXpr<T_ForEach, T_Elem, 1u, stride>(forEach, ptr);
    }

    //lockstep ctxVar
    template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto load(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> const& ctxVar){
        constexpr uint32_t stride = 1u;
        return ReadLeafXpr<T_Foreach<T_Worker, T_Config>, T_Elem, 1u, stride>(forEach, ctxVar);
    }

    template<typename T_ForEach, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_ForEach const& forEach, T_Elem & elem){
        return WriteLeafXpr<T_ForEach, T_Elem, 0u, 0u>(forEach, elem);
    }

    template<typename T_ForEach, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_ForEach const& forEach, T_Elem * const ptr){
        constexpr uint32_t stride = std::decay_t<decltype(forEach.getWorker())>::numWorkers;
        return WriteLeafXpr<T_ForEach, T_Elem, 1u, stride>(forEach, ptr);
    }

    template<template<typename, typename> typename T_Foreach, template<typename, typename> typename T_Variable, typename T_Worker, typename T_Config, typename T_Elem>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto store(T_Foreach<T_Worker, T_Config> const& forEach, T_Variable<T_Elem, T_Config> & ctxVar){
        constexpr uint32_t stride = 1u;
        return WriteLeafXpr<T_Foreach<T_Worker, T_Config>, T_Elem, 1u, stride>(forEach, ctxVar);
    }

//clean up
#undef XPR_OP_WRAPPER
#undef XPR_OP

} // namespace alpaka::lockstep
