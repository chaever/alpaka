///TODO Copyright, License and other stuff here

#pragma once

namespace alpaka::lockstep
{

    //prior to C++ 20 a static operator() is not allowed
#if __cplusplus < 202002L
    #define SIMD_EVAL_F eval
#else
    #define SIMD_EVAL_F operator()
#endif

    struct Addition{
        //should also work for Simd-types that define their own operator+
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left left, T_Right right){
            return left+right;
        }
    };

    template<typename T_Left, typename T_Right, typename T_Functor>
    class Xpr{
        T_Left const& m_leftOperand;
        T_Right const& m_rightOperand;
    public:
        Xpr(T_Left left, T_Right right):m_leftOperand(left), m_rightOperand(right)
        {
        }

        template<typename T_Idx>
        constexpr auto operator[](T_Idx i)
        {
            return T_Functor::SIMD_EVAL_F(m_leftOperand[i], m_rightOperand[i]);
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other){
            using ThisXpr_t = Xpr<T_Left, T_Right, T_Functor>;
            return Xpr<ThisXpr_t, Addition, T_Other>(*this, other);
        }
    };
} // namespace alpaka::lockstep
