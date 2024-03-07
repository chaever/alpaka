///TODO Copyright, License and other stuff here

#pragma once

#include <array>
#include <tuple>


namespace alpaka::lockstep
{
    namespace trait
    {
        namespace detail
        {
            template<typename T>
            struct IsXpr{
                static constexpr bool value = false;
            };

            template<typename T_Functor, typename... T_Operands>
            struct IsXpr<Xpr<T_Functor, T_Operands...>>{
                static constexpr bool value = true;
            };




            template<typename T_Xpr, typename T_Sfinae=void>
            SizeOfSmallestIntermediateResultInByte;

            template<typename T_Xpr, std::enable_if_t<IsXpr<T_Xpr>::value, int> = 0, typename... T_Operands>
            SizeOfSmallestIntermediateResultInByte<T_Xpr>{

                ///TODO assumes that all array-types used have size
                static constexpr auto sizeOfSmallestIntermediateResultInByte = sizeof(T_Array[0u]);

            };

            template<typename T_Functor, typename... T_Operands>
            XprSize<Xpr<T_Functor, T_Operands...>>{

                XprSize

            };
        }















        namespace detail
        {


            template<typename T, typename T_Sfinae=void>
            struct Xpr_t_to_Vector_Container;

            ///TODO is using the
            //1st: for A+b look for operator+, take that return type. -> Problem: there is many operations, not just +
            //2nd: look for leftmost -> most Expressions end with assign -> can just use the type left of the assignment


            //returns leftmost operand vector type (in case of a+b the result is of a's type)
            //only in effect if the 1st operand is not an expression
            template<typename T_Functor, typename T_LeftXpr, typename T_Sfinae=std::enable_if_t<!IsXpr<T_LeftXpr>::value, int> = 0>, typename... T_Operands>
            struct Xpr_t_to_Vector_Container<Xpr<T_Functor, T_LeftXpr, T_Operands...>>{

                //leftmost type in T_Operands
                using type = std::tuple_element_t<0, std::tuple<T_Operands...>>;

            };

            //only in effect when a is an expression in a+b
            template<typename T_Functor, typename T_LeftXpr, typename T_Sfinae=std::enable_if_t<IsXpr<T_LeftXpr>::value, int> = 0>, typename... T_Operands>
            struct Xpr_t_to_Vector_Container<Xpr<T_Functor, T_LeftXpr, T_Operands...>>{

                using type = typename Xpr_t_to_Vector_Container<T_LeftXpr>::type;

            };

        } // namespace detail

        template<typename T_Xpr>
        using Xpr_t_to_Vector_Container_t = Xpr_t_to_Vector_Container<T_Xpr>::type;

        namespace detail
        {

            template<typename T>
            struct IsXpr{
                static constexpr bool value = false;
            };

            template<typename T_Functor, typename... T_Operands>
            struct IsXpr<T_Functor, T_Operands...>{
                static constexpr bool value = true;
            };

            template<typename T, typename T_Sfinae=void>
            struct Xpr_t_to_Vector_Container;

            ///TODO is using the
            //1st: for A+b look for operator+, take that return type. -> Problem: there is many operations, not just +
            //2nd: look for leftmost -> most Expressions end with assign -> can just use the type left of the assignment


            //returns leftmost operand vector type (in case of a+b the result is of a's type)
            //only in effect if the 1st operand is not an expression
            template<typename T_Functor, typename T_LeftXpr, typename T_Sfinae=std::enable_if_t<!IsXpr<T_LeftXpr>::value, int> = 0>, typename... T_Operands>
            struct Xpr_t_to_Vector_Container<Xpr<T_Functor, T_LeftXpr, T_Operands...>>{

                //leftmost type in T_Operands
                using type = std::tuple_element_t<0, std::tuple<T_Operands...>>;

            };

            //only in effect when a is an expression in a+b
            template<typename T_Functor, typename T_LeftXpr, typename T_Sfinae=std::enable_if_t<IsXpr<T_LeftXpr>::value, int> = 0>, typename... T_Operands>
            struct Xpr_t_to_Vector_Container<Xpr<T_Functor, T_LeftXpr, T_Operands...>>{

                using type = typename Xpr_t_to_Vector_Container<T_LeftXpr>::type;

            };

            template<typename T_Xpr>
            using Xpr_t_to_Vector_Container_t = Xpr_t_to_Vector_Container<T_Xpr>::type;

        } // namespace detail





    } // namespace trait
} // namespace alpaka::lockstep
