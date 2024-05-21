/* Copyright 2017-2023 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "alpaka/lockstep/Config.hpp"
#include "alpaka/lockstep/Idx.hpp"
#include "alpaka/lockstep/DeviceCapableArray.hpp"
#include "alpaka/lockstep/Simd.hpp"

#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace lockstep
    {
        template<typename T_Worker, typename T_Config>
        class ForEach;

        /** Variable used by virtual worker
         *
         * This object is designed to hold context variables in lock step
         * programming. A context variable is just a local variable of a virtual
         * worker. Allocating and using a context variable allows to propagate
         * virtual worker states over subsequent lock steps. A context variable
         * for a set of virtual workers is owned by their (physical) worker.
         *
         * Data stored in a context variable should only be used with a lockstep
         * programming construct e.g. lockstep::ForEach<>
         */
        template<typename T_Type, typename T_Config, typename T_SizeInd = T_Type>
        struct Variable
            : protected lockstep::DeviceCapableArray<Pack_t<T_Type, T_SizeInd>, alpaka::core::divCeil(T_Config::domainSize, T_Config::numWorkers * laneCount_v<Pack_t<T_Type, T_SizeInd>>)>
            , T_Config
        {
            using T_Config::domainSize;
            using T_Config::numWorkers;
            using T_Config::simdSize;

            //in simd packs
            constexpr static auto numSimdPacks = alpaka::core::divCeil(T_Config::domainSize, T_Config::numWorkers * laneCount_v<Pack_t<T_Type, T_SizeInd>>);

            using BaseArray = lockstep::DeviceCapableArray<Pack_t<T_Type, T_SizeInd>, numSimdPacks>;

            using pack_t = typename BaseArray::value_type;

            using value_type = elemTOfPack_t<pack_t>;

            /** default constructor
             *
             * Data member are uninitialized.
             * This method must be called collectively by all workers.
             */
            Variable() = default;

            /** constructor
             *
             * Initialize each member with the given value.
             * This method must be called collectively by all workers.
             *
             * @param args element assigned to each member
             */
            template<typename... T_Args>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE explicit Variable(T_Args&&... args) : BaseArray(std::forward<T_Args>(args)...)
            {
            }

            /** disable copy constructor
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable(Variable const&) = delete;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable(Variable&&) = default;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable& operator=(Variable&&) = default;

             /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE value_type const& operator[](Idx const idx) const
            {
                constexpr auto laneCount = laneCount_v<pack_t>;
                return getElem(BaseArray::operator[](idx.getWorkerElemIdx()/laneCount), idx.getWorkerElemIdx()%laneCount);
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE value_type & operator[](Idx const idx)
            {
                constexpr auto laneCount = laneCount_v<pack_t>;
                return getElem(BaseArray::operator[](idx.getWorkerElemIdx()/laneCount), idx.getWorkerElemIdx()%laneCount);
            }
            /** @} */

            //const access to packs
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE pack_t const& packAt(uint32_t const idx) const
            {
                return BaseArray::operator[](idx);
            }

            //non-const access to packs
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE pack_t& packAt(uint32_t const idx)
            {
                return BaseArray::operator[](idx);
            }

#define OPERATOR() operator
#define OPERATOR_DEF_VAR_ASSIGN(op)\
            /*var op var*/\
            template<typename T_Other_Type, typename T_Other_SizeInd>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Other_Type, T_Config, T_Other_SizeInd> const other){\
                static_assert(laneCount_v<pack_t> == laneCount_v<typename Variable<T_Other_Type, T_Config, T_Other_SizeInd>::pack_t>);\
                for(auto i=0u; i<numSimdPacks; ++i){\
                    packAt(i) op other.packAt(i);\
                }\
            }\
            /*var op scalar*/\
            template<typename T_Other_Type>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (T_Other_Type const other){\
                static_assert(std::is_arithmetic_v<std::decay_t<T_Other_Type>>);\
                for(auto i=0u; i<numSimdPacks; ++i){\
                    packAt(i) op other;\
                }\
            }

            OPERATOR_DEF_VAR_ASSIGN(=)
            OPERATOR_DEF_VAR_ASSIGN(+=)
            OPERATOR_DEF_VAR_ASSIGN(-=)

        }; // Variable struct

#define OPERATOR_DEF_VAR_BINARY(op)\
        /*var op var*/\
        template<typename T_Type_Left, typename T_SizeInd_Left, typename T_Config, typename T_Type_Right, typename T_SizeInd_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Type_Left, T_Config, T_SizeInd_Left> const left, Variable<T_Type_Right, T_Config, T_SizeInd_Right> const right){\
            using result_elem_t = decltype(std::declval<T_Type_Left> op std::declval<T_Type_Right>);\
            using left_t = std::decay_t<decltype(left)>;\
            using right_t = std::decay_t<decltype(right)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<typename left_t::pack_t>, std::decay_t<typename right_t::pack_t>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            /*make sure that elemCount of packs matches*/\
            static_assert(laneCount_v<typename left_t::pack_t> == laneCount_v<typename right_t::pack_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<left_t::numSimdPacks; ++i){\
                tmp.packAt(i) = left.packAt(i) op right.packAt(i);\
            }\
            return tmp;\
        }\
        /*var op scalar*/\
        template<typename T_Type_Left, typename T_SizeInd_Left, typename T_Config, typename T_Type_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Type_Left, T_Config, T_SizeInd_Left> const left, T_Type_Right const right){\
            static_assert(std::is_arithmetic_v<std::decay_t<T_Type_Right>>);\
            using result_elem_t = decltype(std::declval<T_Type_Left> op std::declval<T_Type_Right>);\
            using left_t = std::decay_t<decltype(left)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<typename left_t::pack_t>, std::decay_t<T_Type_Right>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<left_t::numSimdPacks; ++i){\
                tmp.packAt(i) = left.packAt(i) op right;\
            }\
            return tmp;\
        }\
        /*scalar op var*/\
        template<typename T_Type_Left, typename T_Type_Right, typename T_Config, typename T_SizeInd_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (T_Type_Left const left, Variable<T_Type_Right, T_Config, T_SizeInd_Right> const right){\
            static_assert(std::is_arithmetic_v<std::decay_t<T_Type_Left>>);\
            using result_elem_t = decltype(std::declval<T_Type_Left> op std::declval<T_Type_Right>);\
            using right_t = std::decay_t<decltype(right)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<T_Type_Left>, std::decay_t<typename right_t::pack_t>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<right_t::numSimdPacks; ++i){\
                tmp.packAt(i) = left op right.packAt(i);\
            }\
            return tmp;\
        }

#define OPERATOR_DEF_VAR_PREFIX(op)\
        template<typename T_Type, typename T_Config, typename T_SizeInd>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Type, T_Config, T_SizeInd> var){\
            /*we assume that all unary operations eturn the same type, make sure that this assumption is correct*/\
            static_assert(std::is_same_v<T_Type, decltype(op std::declval<T_Type>)>);\
            for(auto i=0u; i<Variable<T_Type, T_Config, T_SizeInd>::numSimdPacks; ++i){\
                var.packAt(i) = op var.packAt(i);\
            }\
            return var;\
        }

    ///TODO needs to be defined outside the namespace!
#define FREE_FUNC_DEF_VAR(funcName)\
        template<typename T_Type, typename T_Config, typename T_SizeInd>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) fName(alpaka::lockstep::Variable<T_Type, T_Config, T_SizeInd> var){\
            using ret_t = decltype(fName(var[std::declval<std::uint32_t>]));\
            static_assert(std::is_same_v<T_Type, ret_t>);\
            for(auto i=0u; i<alpaka::lockstep::Variable<T_Type, T_Config, T_SizeInd>::numSimdPacks; ++i){\
                var.packAt(i) = fName(var.packAt(i));\
            }\
            return var;\
        }

        OPERATOR_DEF_VAR_BINARY(+)
        OPERATOR_DEF_VAR_BINARY(-)
        OPERATOR_DEF_VAR_BINARY(*)
        OPERATOR_DEF_VAR_BINARY(/)
        OPERATOR_DEF_VAR_BINARY(%)
        OPERATOR_DEF_VAR_BINARY(<<)
        OPERATOR_DEF_VAR_BINARY(>>)

        OPERATOR_DEF_VAR_PREFIX(!)
        OPERATOR_DEF_VAR_PREFIX(~)

    } // namespace lockstep
} // namespace alpaka

        FREE_FUNC_DEF_VAR(std::abs)

namespace alpaka
{
    namespace lockstep
    {
        ///TODO need shift operators -> var << var, var << int




#undef OPERATOR_DEF_VAR_ASSIGN
#undef FREE_FUNC_DEF_VAR
#undef OPERATOR_DEF_VAR_PREFIX
#undef OPERATOR_DEF_VAR_BINARY
#undef OPERATOR

        /** Creates a variable usable within a lockstep step
         *
         * @attention: Data is uninitialized.
         *
         * @tparam T_Type type of the variable
         * @tparam T_Config lockstep config
         *
         * @param forEach Lockstep for each algorithm to derive the required memory for the variable.
         * @return Variable usable within a lockstep step. Variable data can not be accessed outside of a lockstep
         * step.
         */
        template<typename T_Type, typename T_Worker, typename T_Config>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto makeVar(ForEach<T_Worker, T_Config> const& forEach)
        {
            return Variable<T_Type, typename ForEach<T_Worker, T_Config>::BaseConfig>();
        }

        /** Creates a variable usable within a subsequent locksteps.
         *
         * Constructor will be called with the given arguments T_Args.
         * @attention The constructor should not contain a counter to count the number of constructor invocations. The
         * number of invocations can be larger than the number of indices in the lockstep domain.
         *
         * @tparam T_Type type of the variable
         * @tparam T_Config lockstep config
         * @tparam T_Args type of the constructor arguments
         *
         * @param forEach Lockstep for each algorithm to derive the required memory for the variable.
         * @param args Arguments passed to the constructor of the variable.
         * @return Variable usable within a lockstep step. Variable data can not be accessed outside of a lockstep
         * step.
         */
        template<typename T_Type, typename T_Worker, typename T_Config, typename... T_Args>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto makeVar(ForEach<T_Worker, T_Config> const& forEach, T_Args&&... args)
        {
            return Variable<T_Type, typename ForEach<T_Worker, T_Config>::BaseConfig>(std::forward<T_Args>(args)...);
        }

    } // namespace lockstep
} // namespace alpaka
