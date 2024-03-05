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

#include "Simd.hpp"
#include "ExpressionTemplates.hpp"

namespace alpaka
{
    namespace lockstep
    {
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
                static const T_Elem& eval(T_Idx idx, T_Elem const * const ptr)
                {
                    return ptr[idx];
                }
            };

            //specialization for SIMD-SimdLookupIndex
            //returns only const Packs because they are copies
            template<typename T_Type>
            struct IndexOperator<SimdLookupIndex<T_Type>>{

                template<typename T_Elem>
                static SimdPack_t<T_Elem> eval(SimdLookupIndex<T_Type> idx, T_Elem* const ptr)
                {
                    static_assert(std::is_same_v<T_Type, T_Elem>);
                    SimdPack_t<T_Elem> tmp;
                    tmp.loadUnaligned(ptr + static_cast<uint32_t>(idx));
                    return tmp;
                }

                template<typename T_Elem>
                static const SimdPack_t<T_Elem> eval(SimdLookupIndex<T_Type> idx, T_Elem const * const ptr)
                {
                    static_assert(std::is_same_v<T_Type, T_Elem>);
                    SimdPack_t<T_Elem> tmp;
                    tmp.loadUnaligned(ptr + static_cast<uint32_t>(idx));
                    return tmp;
                }
            };
        } // namespace detail




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
                return data()[idx];
            }

            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE reference operator[](T_Idx const idx)
            {
                return data()[idx];
            }
            /** @} */

            //used by alpaka::lockstep::Xpr for the evaluation of Expression objects
            template<typename T_Idx>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const auto getValueAtIndex(T_Idx const idx) const
            {
                return detail::IndexOperator<T_Idx>::eval(idx, &this[0]);
            }

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
