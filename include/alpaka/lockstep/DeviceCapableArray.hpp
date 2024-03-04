/* Copyright 2016-2023 Rene Widera
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

namespace alpaka
{
    namespace lockstep
    {
        /** static sized array
         *
         * mimic the most parts of the `std::array`
         */
        template<typename T_Type, size_t T_size>
        struct DeviceCapableArray
        {
            using value_type = T_Type;
            using size_type = size_t;
            using reference = value_type&;
            using const_reference = value_type const&;
            using pointer = value_type*;
            using const_pointer = value_type const*;

            /** get number of elements */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
            constexpr size_type size() const
            {
                return T_size;
            }

            /** get maximum number of elements */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
            constexpr size_type max_size() const
            {
                return T_size;
            }

            /** get the direct access to the internal data
             *
             * @{
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
            pointer data()
            {
                return reinterpret_cast<pointer>(m_data);
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
            const_pointer data() const
            {
                return reinterpret_cast<const_pointer>(m_data);
            }
            /** @} */

            /** default constructor
             *
             * all members are uninitialized
             */
            DeviceCapableArray() = default;

            /** constructor
             *
             * initialize each member with the given value
             *
             * @param value element assigned to each member
             */
            template<typename... T_Args>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE DeviceCapableArray(T_Args&&... args)
            {
                for(size_type i = 0; i < size(); ++i)
                    reinterpret_cast<T_Type*>(m_data)[i] = std::move(T_Type{std::forward<T_Args>(args)...});
            }

            /** get N-th value
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const_reference operator[](T_Idx const idx) const
            {
                return reinterpret_cast<T_Type const*>(m_data)[idx];
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE reference operator[](T_Idx const idx)
            {
                return reinterpret_cast<T_Type*>(m_data)[idx];
            }
            /** @} */

            template<typename T_Xpr>
            constexpr auto operator=(T_Xpr const&);

            template<typename T_Other>
            constexpr auto operator+(T_Other const &);

        private:
            /** data storage
             *
             * std::array is a so-called "aggregate" which does not default-initialize
             * its members. In order to allow arbitrary types to skip implementing
             * a default constructur, this member is not stored as
             * `value_type m_data[ T_size ]` but as type-size aligned Byte type.
             */
            uint8_t m_data alignas(alignof(T_Type))[T_size * sizeof(T_Type)];
        };

    } // namespace lockstep
} // namespace alpaka
