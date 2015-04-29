/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

// Base classes.
#include <alpaka/core/BasicWorkDiv.hpp>     // workdiv::BasicWorkDiv
#include <alpaka/accs/serial/Idx.hpp>       // IdxSerial
#include <alpaka/accs/serial/Atomic.hpp>    // AtomicSerial

// Specialized traits.
#include <alpaka/traits/Acc.hpp>            // AccType
#include <alpaka/traits/Exec.hpp>           // ExecType
#include <alpaka/traits/Event.hpp>          // EventType
#include <alpaka/traits/Dev.hpp>            // DevType
#include <alpaka/traits/Stream.hpp>         // StreamType

// Implementation details.
#include <alpaka/devs/cpu/Dev.hpp>          // DevCpu
#include <alpaka/devs/cpu/Event.hpp>        // EventCpu
#include <alpaka/devs/cpu/Stream.hpp>       // StreamCpu

#include <boost/core/ignore_unused.hpp>     // boost::ignore_unused

#include <memory>                           // std::unique_ptr
#include <vector>                           // std::vector

namespace alpaka
{
    namespace accs
    {
        //-----------------------------------------------------------------------------
        //! The serial accelerator.
        //-----------------------------------------------------------------------------
        namespace serial
        {
            //-----------------------------------------------------------------------------
            //! The serial accelerator implementation details.
            //-----------------------------------------------------------------------------
            namespace detail
            {
                // Forward declaration.
                class ExecSerial;

                //#############################################################################
                //! The serial accelerator.
                //!
                //! This accelerator allows serial kernel execution on a cpu device.
                //! The block size is restricted to 1x1x1 so there is no parallelism at all.
                //#############################################################################
                class AccSerial :
                    protected alpaka::workdiv::BasicWorkDiv,
                    protected IdxSerial,
                    protected AtomicSerial
                {
                public:
                    friend class ::alpaka::accs::serial::detail::ExecSerial;

                private:
                    //-----------------------------------------------------------------------------
                    //! Constructor.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TWorkDiv>
                    ALPAKA_FCT_ACC_NO_CUDA AccSerial(
                        TWorkDiv const & workDiv) :
                            alpaka::workdiv::BasicWorkDiv(workDiv),
                            IdxSerial(m_v3uiGridBlockIdx),
                            AtomicSerial(),
                            m_v3uiGridBlockIdx(Vec3<>::zeros())
                    {}

                public:
                    //-----------------------------------------------------------------------------
                    //! Copy constructor.
                    // Do not copy most members because they are initialized by the executor for each execution.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccSerial(AccSerial const &) = delete;
    #if (!BOOST_COMP_MSVC) || (BOOST_COMP_MSVC >= BOOST_VERSION_NUMBER(14, 0, 0))
                    //-----------------------------------------------------------------------------
                    //! Move constructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA AccSerial(AccSerial &&) = delete;
    #endif
                    //-----------------------------------------------------------------------------
                    //! Copy assignment.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA auto operator=(AccSerial const &) -> AccSerial & = delete;
                    //-----------------------------------------------------------------------------
                    //! Destructor.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA virtual ~AccSerial() noexcept = default;

                    //-----------------------------------------------------------------------------
                    //! \return The requested indices.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getIdx() const
                    -> Vec<TDim>
                    {
                        return idx::getIdx<TOrigin, TUnit, TDim>(
                            *static_cast<IdxSerial const *>(this),
                            *static_cast<alpaka::workdiv::BasicWorkDiv const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The requested extents.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOrigin,
                        typename TUnit,
                        typename TDim = dim::Dim3>
                    ALPAKA_FCT_ACC_NO_CUDA auto getWorkDiv() const
                    -> Vec<TDim>
                    {
                        return workdiv::getWorkDiv<TOrigin, TUnit, TDim>(
                            *static_cast<alpaka::workdiv::BasicWorkDiv const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Execute the atomic operation on the given address with the given value.
                    //! \return The old value before executing the atomic operation.
                    //-----------------------------------------------------------------------------
                    template<
                        typename TOp,
                        typename T>
                    ALPAKA_FCT_ACC auto atomicOp(
                        T * const addr,
                        T const & value) const
                    -> T
                    {
                        return atomic::atomicOp<TOp, T>(
                            addr,
                            value,
                            *static_cast<AtomicSerial const *>(this));
                    }

                    //-----------------------------------------------------------------------------
                    //! Syncs all threads in the current block.
                    //-----------------------------------------------------------------------------
                    ALPAKA_FCT_ACC_NO_CUDA void syncBlockThreads() const
                    {
                        // Nothing to do in here because only one thread in a group is allowed.
                    }

                    //-----------------------------------------------------------------------------
                    //! \return Allocates block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T,
                        UInt TuiNumElements>
                    ALPAKA_FCT_ACC_NO_CUDA auto allocBlockSharedMem() const
                    -> T *
                    {
                        static_assert(TuiNumElements > 0, "The number of elements to allocate in block shared memory must not be zero!");

                        // \TODO: C++14 std::make_unique would be better.
                        m_vvuiSharedMem.emplace_back(
                            std::unique_ptr<uint8_t[]>(
                                reinterpret_cast<uint8_t*>(new T[TuiNumElements])));
                        return reinterpret_cast<T*>(m_vvuiSharedMem.back().get());
                    }

                    //-----------------------------------------------------------------------------
                    //! \return The pointer to the externally allocated block shared memory.
                    //-----------------------------------------------------------------------------
                    template<
                        typename T>
                    ALPAKA_FCT_ACC_NO_CUDA auto getBlockSharedExternMem() const
                    -> T *
                    {
                        return reinterpret_cast<T*>(m_vuiExternalSharedMem.get());
                    }

    #ifdef ALPAKA_NVCC_FRIEND_ACCESS_BUG
                protected:
    #else
                private:
    #endif
                    // getIdx
                    Vec3<> mutable m_v3uiGridBlockIdx;                         //!< The index of the currently executed block.

                    // allocBlockSharedMem
                    std::vector<
                        std::unique_ptr<uint8_t[]>> mutable m_vvuiSharedMem;    //!< Block shared memory.

                    // getBlockSharedExternMem
                    std::unique_ptr<uint8_t[]> mutable m_vuiExternalSharedMem;  //!< External block shared memory.
                };
            }
        }
    }

    using AccSerial = accs::serial::detail::AccSerial;

    namespace traits
    {
        namespace acc
        {
            //#############################################################################
            //! The serial accelerator accelerator type trait specialization.
            //#############################################################################
            template<>
            struct AccType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::AccSerial;
            };
            //#############################################################################
            //! The serial accelerator device properties get trait specialization.
            //#############################################################################
            template<>
            struct GetAccDevProps<
                accs::serial::detail::AccSerial>
            {
                ALPAKA_FCT_HOST static auto getAccDevProps(
                    devs::cpu::detail::DevCpu const & dev)
                -> alpaka::acc::AccDevProps
                {
                    boost::ignore_unused(dev);

                    return alpaka::acc::AccDevProps(
                        // m_uiMultiProcessorCount
                        1u,
                        // m_uiBlockThreadsCountMax
                        1u,
                        // m_v3uiBlockThreadExtentsMax
                        Vec3<>::ones(),
                        // m_v3uiGridBlockExtentsMax
                        Vec3<>::all(std::numeric_limits<Vec3<>::Val>::max()));
                }
            };
            //#############################################################################
            //! The serial accelerator name trait specialization.
            //#############################################################################
            template<>
            struct GetAccName<
                accs::serial::detail::AccSerial>
            {
                ALPAKA_FCT_HOST_ACC static auto getAccName()
                -> std::string
                {
                    return "AccSerial";
                }
            };
        }

        namespace event
        {
            //#############################################################################
            //! The serial accelerator event type trait specialization.
            //#############################################################################
            template<>
            struct EventType<
                accs::serial::detail::AccSerial>
            {
                using type = devs::cpu::detail::EventCpu;
            };
        }

        namespace exec
        {
            //#############################################################################
            //! The serial accelerator executor type trait specialization.
            //#############################################################################
            template<>
            struct ExecType<
                accs::serial::detail::AccSerial>
            {
                using type = accs::serial::detail::ExecSerial;
            };
        }

        namespace dev
        {
            //#############################################################################
            //! The serial accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevType<
                accs::serial::detail::AccSerial>
            {
                using type = devs::cpu::detail::DevCpu;
            };
            //#############################################################################
            //! The serial accelerator device type trait specialization.
            //#############################################################################
            template<>
            struct DevManType<
                accs::serial::detail::AccSerial>
            {
                using type = devs::cpu::detail::DevManCpu;
            };
        }

        namespace stream
        {
            //#############################################################################
            //! The serial accelerator stream type trait specialization.
            //#############################################################################
            template<>
            struct StreamType<
                accs::serial::detail::AccSerial>
            {
                using type = devs::cpu::detail::StreamCpu;
            };
        }
    }
}
