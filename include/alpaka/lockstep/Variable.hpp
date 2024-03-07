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


#include <type_traits>
#include <utility>

namespace alpaka
{
    namespace lockstep
    {
        template<typename T_Worker, typename T_Config>
        class ForEach;

        namespace detail
        {

            template<typename T_Idx>
            struct IndexOperator{

                template<typename T_Elem>
                static T_Elem& eval(T_Idx idx, T_Elem* const ptr)
                {
                    return ptr[idx];
                }

                template<typename T_Elem>
                static T_Elem const& eval(T_Idx idx, T_Elem const * const ptr)
                {
                    return ptr[idx];
                }
            };

            //specialization for SIMD-SimdLookupIndex
            //returns only const Packs because they are copies
            template<typename T_Type>
            struct IndexOperator<SimdLookupIndex<T_Type>>{

                static const Pack_t<T_Type> eval(SimdLookupIndex<T_Type> idx, T_Type const * const ptr)
                {
                    return SimdInterface_t<T_Type>::loadUnaligned(ptr + static_cast<uint32_t>(idx));;
                }
            };
        } // namespace detail

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
        template<typename T_Type, typename T_Config>
        struct Variable
            : protected lockstep::DeviceCapableArray<T_Type, T_Config::maxIndicesPerWorker>
            , T_Config
        {
            using T_Config::domainSize;
            using T_Config::numWorkers;
            using T_Config::simdSize;

            using BaseArray = lockstep::DeviceCapableArray<T_Type, T_Config::maxIndicesPerWorker>;

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

            //extend Variable to allow Xpr assignment
            template<typename T_Left, typename T_Right, typename T_Functor>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto& operator=(lockstep::ReadXpr<T_Functor, T_Left, T_Right> const& xpr) {

                constexpr auto lanes = laneCount<T_Type>;
                constexpr auto vectorLoops = T_Config::maxIndicesPerWorker/lanes;

                //get pointer to start of internal data storage
                T_Type* ptr = BaseArray::data();

                for(std::size_t i = 0u; i<vectorLoops; ++i, ptr+=lanes){
                    //uses the getValueAtIndex that returns Pack_t
                    SimdInterface_t<T_Type>::storeUnaligned(xpr.getValueAtIndex(SimdLookupIndex<T_Type>(i)), ptr);
                }
                for(std::size_t i = vectorLoops*lanes; i<T_Config::maxIndicesPerWorker; ++i, ++ptr){
                    //uses the getValueAtIndex that returns T_Type
                    *ptr = xpr.getValueAtIndex(i);
                }
                return *this;
            }

            //defines Variable + {Variable or ReadXpr}
            template<typename T_Other>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto operator+(T_Other const & other) {
                using ThisVar_t = Variable<T_Type, T_Config>;
                return ReadXpr<Addition, ThisVar_t, T_Other>(*this, other);
            }

            /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE typename BaseArray::const_reference operator[](Idx const idx) const
            {
                return BaseArray::operator[](idx.getWorkerElemIdx());
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE typename BaseArray::reference operator[](Idx const idx)
            {
                return BaseArray::operator[](idx.getWorkerElemIdx());
            }
            /** @} */


            //used by alpaka::lockstep::ReadXpr for the evaluation of Expression objects
            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto getValueAtIndex(T_Idx const idx)
            {
                return detail::IndexOperator<T_Idx>::eval(idx, BaseArray::data());
            }
            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const auto getValueAtIndex(T_Idx const idx) const
            {
                return detail::IndexOperator<T_Idx>::eval(idx, BaseArray::data());
            }

        };

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
