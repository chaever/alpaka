/* Copyright 2022-2023 Rene Widera
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

#include "alpaka/lockstep/Worker.hpp"
#include "GetNumWorkers.hpp"


namespace alpaka::lockstep
{
    namespace detail
    {
        /** Helper to manage visibility for methods in WorkerCfg.
         *
         * This object provides an optimized access to the one dimensional worker index of WorkerCfg.
         * This indirection avoids that the user is accessing the optimized method from a kernel which is not
         * guaranteed launched with a one dimensional block size.
         */
        struct WorkerCfgAssume1DBlock
        {
            /** Get the lockstep worker index.
             *
             * @attention This method should only be called if it is guaranteed that the kernel is started with a one
             * dimension block size. In general this method should only be used from
             * lockstep::exec::detail::LockStepKernel.
             *
             * @tparam T_WorkerCfg lockstep worker configuration
             * @tparam T_Acc alpaka accelerator type
             * @param acc alpaka accelerator
             * @return worker index
             */
            template<typename T_WorkerCfg, typename T_Acc>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getWorker(T_Acc const& acc)
            {
                return T_WorkerCfg::getWorkerAssume1DThreads(acc);
            }
        };
    } // namespace detail
    /** Configuration of worker used for a lockstep kernel
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers. Do not assume that the suggested number of
     *                               workers is used within the kernel. The real used number of worker can be queried
     *                               with getNumWorkers() or via the member variable numWorkers.
     *
     * @attention: The real number of workers used for the lockstep kernel depends on the alpaka backend and will
     * be adjusted by this class via the trait alpaka::traits::GetNumWorkers.
     */
    template<uint32_t T_numSuggestedWorkers>
    struct WorkerCfg
    {
        friend struct detail::WorkerCfgAssume1DBlock;

        /** adjusted number of workers
         *
         * This number is taking the block size restriction of the alpaka backend into account.
         */
        template<typename T_Acc>
        static constexpr uint32_t numWorkers = alpaka::traits::GetNumWorkers<T_numSuggestedWorkers, AccToTag<T_Acc>>::value;

        /** get the worker index
         *
         * @return index of the worker
         */
        template<typename T_Acc>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getWorker(T_Acc const& acc)
        {
            auto const localThreadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
            auto const blockExtent = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

            // validate that the kernel is started with the correct number of threads
            ALPAKA_ASSERT_OFFLOAD(blockExtent.prod() == numWorkers<T_Acc>);

            auto const linearThreadIdx = alpaka::mapIdx<1u>(localThreadIdx, blockExtent)[0];
            return Worker<T_Acc, T_numSuggestedWorkers>(acc, linearThreadIdx);
        }

        /** get the number of workers
         *
         * @tparam T_Acc alpaka accelerator type
         * @return number of workers
         */
        template<typename T_Acc>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static constexpr uint32_t getNumWorkers()
        {
            return numWorkers<T_Acc>;
        }

    private:
        /** Get the lockstep worker index.
         *
         * @tparam T_Acc alpaka accelerator type
         * @param acc alpaka accelerator
         * @return lockstep worker index
         */
        template<typename T_Acc>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE static auto getWorkerAssume1DThreads(T_Acc const& acc)
        {
            [[maybe_unused]] auto const blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc).x;

            // validate that the kernel is started with the correct number of threads
            ALPAKA_ASSERT_OFFLOAD(blockDim == numWorkers<T_Acc>);

            return Worker<T_Acc, T_numSuggestedWorkers>(acc, alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc).x);
        }
    };

    /** Creates a lockstep worker configuration.
     *
     * @tparam T_numSuggestedWorkers Suggested number of lockstep workers.
     * @return lockstep worker configuration
     */
    template<uint32_t T_numSuggestedWorkers>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto makeWorkerCfg()
    {
        return WorkerCfg<T_numSuggestedWorkers>{};
    }
} // namespace alpaka::lockstep
