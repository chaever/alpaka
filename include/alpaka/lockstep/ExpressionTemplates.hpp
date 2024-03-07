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
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){
            //uses T_Left::operator+(T_Right)
            return left+right;
        }
    };

    struct Assignment{
        //should also work for Simd-types that define their own operator+
        template<typename T_Left, typename T_Right, std::enable_if_t<!std::is_same_v<T_Right, Pack_t>, int> = 0>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
            //uses T_Left::operator=(T_Right)
            return left=right;
        }
        template<typename T_Left, typename T_Right, std::enable_if_t< std::is_same_v<T_Right, Pack_t>, int> = 0>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){

            //uses T_Left::operator=(T_Right)
            return right.copy_to(&left);
        }
    };

    //const left operand
    template<typename T_Left, typename T_Right, typename T_Functor>
    class Xpr{
        T_Left const& m_leftOperand;
        T_Right const& m_rightOperand;
    public:
        Xpr(T_Left const& left, T_Right const& right):m_leftOperand(left), m_rightOperand(right)
        {
        }

        using ThisXpr_t = Xpr<T_Left, T_Right, T_Functor>;

        template<typename T_Idx>
        constexpr auto const getValueAtIndex(T_Idx const i) const
        {
            return T_Functor::SIMD_EVAL_F(m_leftOperand.getValueAtIndex(i), m_rightOperand.getValueAtIndex(i));
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            return Xpr<ThisXpr_t, T_Other, Addition>(*this, other);
        }

        //writing to a readOnly expression is forbidden
        template<typename T_Other>
        constexpr auto operator=(T_Other const & other) = delete;
    };

    //non-const left operand to write to
    template<typename T_Left, typename T_Right, typename T_Functor>
    class WriteableXpr{
        T_Left & m_leftOperand;
        T_Right const& m_rightOperand;
    public:
        WriteableXpr(T_Left & left, T_Right const& right):m_leftOperand(left), m_rightOperand(right)
        {
        }

        using ThisXpr_t = WriteableXpr<T_Left, T_Right, T_Functor>;

        template<typename T_Idx>
        constexpr auto const getValueAtIndex(T_Idx const i) const
        {
            //operator[] returns reference, which is then assignable
            //the cast transforms any SimdLookupIndices into ints
            return Assignment::SIMD_EVAL_F(m_leftOperand[static_cast<uint32_t>(i)], m_rightOperand.getValueAtIndex(i));
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            return Xpr<ThisXpr_t, T_Other, Addition>(*this, other);
        }

        template<typename T_Other>
        constexpr auto operator=(T_Other const & other) const
        {
            return WriteableXpr<ThisXpr_t, T_Other, Assignment>(*this, other);
        }
    };
} // namespace alpaka::lockstep
