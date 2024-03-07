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

    namespace detail
    {
        template<typename T_Idx>
        struct IndexOperatorLeafRead{

            template<typename T_Elem>
            static T_Elem const& eval(T_Idx idx, T_Elem const * const ptr)
            {
                return ptr[idx];
            }
        };

        //specialization for SIMD-SimdLookupIndex
        //returns only const Packs because they are copies
        template<typename T_Type>
        struct IndexOperatorLeafRead<SimdLookupIndex<T_Type>>{

            static const Pack_t<T_Type> eval(SimdLookupIndex<T_Type> idx, T_Type const * const ptr)
            {
                return SimdInterface_t<T_Type>::loadUnaligned(ptr + static_cast<uint32_t>(idx));
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
            static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, Pack_t<T_Left> const& right){
                SimdInterface_t<T_Left>::storeUnaligned(right, &left);
                return right;//return right to preserve semantics of operator=
            }
        };
    } // namespace detail

    struct Addition{
        //should also work for Simd-types that define their own operator+
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left const& left, T_Right const& right){
            //uses T_Left::operator+(T_Right)
            return left+right;
        }
    };

    struct Assignment{
        template<typename T_Left, typename T_Right>
        static constexpr decltype(auto) SIMD_EVAL_F(T_Left& left, T_Right const& right){
            return detail::AssignmentTrait<T_Left, T_Right>::SIMD_EVAL_F(left, right);
        }
    };

    //forward declarations
    template<typename T_Functor, typename T_Left, typename T_Right>
    class ReadXpr;
    template<typename T_Functor, typename T_Left, typename T_Right>
    class WriteableXpr;

    //cannot be assigned to
    //can be made from pointers, or some container classes
    template<typename T>
    class ReadLeafXpr{
        T const& m_source;
    public:
        using ThisXpr_t = ReadLeafXpr<T>;
        ///TODO maybe this class should know the size of the array it points to?

        ReadLeafXpr(T const& source) : m_source(source)
        {
        }

        //allows making an expression from CtxVariable
        template<typename T_Config>
        ReadLeafXpr(lockstep::Variable<T, T_Config> const& v) : m_source(v[lockstep::Idx(0,0)])
        {
        }

        //returns const ref, since Leaves are not meant to be assigned to
        template<typename T_Idx>
        auto const& operator[](T_Idx const idx) const
        {
            return detail::IndexOperatorLeafRead<T_Idx>::eval(idx, &m_source);
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            return ReadXpr<Addition, ThisXpr_t, T_Other>(*this, other);
        }
    };

    //can be assigned to
    //can be made from pointers, or some container classes
    template<typename T>
    class WriteLeafXpr{
        T & m_dest;
    public:
        using ThisXpr_t = WriteLeafXpr<T>;
        ///TODO maybe this class should know the size of the array it points to?

        WriteLeafXpr(T & dest) : m_dest(dest)
        {
        }

        //allows making an expression from CtxVariable
        template<typename T_Config>
        WriteLeafXpr(lockstep::Variable<T, T_Config> & v) : m_dest(v[lockstep::Idx(0,0)])
        {
        }

        //returns ref to allow assignment
        template<typename T_Idx>
        auto & operator[](T_Idx const idx)
        {
            //static_cast will turn SimdLookupIndex into a flattened value when required
            return (&m_dest)[static_cast<uint32_t>(idx)];
        }

        template<typename T_Other>
        constexpr decltype(auto) operator=(T_Other const & other)
        {
            return WriteableXpr<Assignment, ThisXpr_t, T_Other>(*this, other);
        }
    };

    //const left operand, cannot assign
    template<typename T_Functor, typename T_Left, typename T_Right>
    class ReadXpr{
        T_Left const& m_leftOperand;
        T_Right const& m_rightOperand;
    public:
        ReadXpr(T_Left const& left, T_Right const& right):m_leftOperand(left), m_rightOperand(right)
        {
        }

        using ThisXpr_t = ReadXpr<T_Functor, T_Left, T_Right>;

        template<typename T_Idx>
        constexpr auto const getValueAtIndex(T_Idx const i) const
        {
            return T_Functor::SIMD_EVAL_F(m_leftOperand.getValueAtIndex(i), m_rightOperand.getValueAtIndex(i));
        }

        template<typename T_Other>
        constexpr auto operator+(T_Other const & other) const
        {
            return ReadXpr<Addition, ThisXpr_t, T_Other>(*this, other);
        }

        //writing to a readOnly expression is forbidden
        template<typename T_Other>
        constexpr auto operator=(T_Other const & other) = delete;
    };

    //non-const left operand to write to
    template<typename T_Functor, typename T_Left, typename T_Right>
    class WriteableXpr{
        T_Left & m_leftOperand;
        T_Right const& m_rightOperand;
    public:
        WriteableXpr(T_Left & left, T_Right const& right):m_leftOperand(left), m_rightOperand(right)
        {
        }

        using ThisXpr_t = WriteableXpr<T_Functor, T_Left, T_Right>;

        template<typename T_Idx>
        constexpr auto const getValueAtIndex(T_Idx const i) const
        {
            //operator[] returns reference, which is then assignable
            //the cast transforms any SimdLookupIndices into ints
            return T_Functor::SIMD_EVAL_F(m_leftOperand.getValueAtIndex(i), m_rightOperand.getValueAtIndex(i));
        }

        template<typename T_Other>
        constexpr auto operator=(T_Other const & other) const
        {
            return WriteableXpr<Assignment, ThisXpr_t, T_Other>(*this, other);
        }
    };
} // namespace alpaka::lockstep
