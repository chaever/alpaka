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



#include <type_traits>

namespace alpaka
{
    namespace trait
    {
        /** Get number of workers
         *
         * the number of workers for a kernel depending on the used accelerator
         *
         * @tparam T_maxWorkers the maximum number of workers
         * @tparam T_Acc the accelerator type
         * @return @p ::value number of workers
         */
        template<uint32_t T_maxWorkers, typename T_AccTag>
        struct GetNumWorkers
        {
            static constexpr uint32_t value = T_maxWorkers;
        };

        template<uint32_t T_maxWorkers>
        struct GetNumWorkers<T_maxWorkers, alpaka::TagCpuOmp2Blocks>
        {
            static constexpr uint32_t value = 1u;
        };

        template<uint32_t T_maxWorkers>
        struct GetNumWorkers<T_maxWorkers, alpaka::TagCpuSerial>
        {
            static constexpr uint32_t value = 1u;
        };

        template<uint32_t T_maxWorkers>
        struct GetNumWorkers<T_maxWorkers, alpaka::TagCpuTbbBlocks>
        {
            static constexpr uint32_t value = 1u;
        };
    } // namespace trait
} // namespace alpaka


