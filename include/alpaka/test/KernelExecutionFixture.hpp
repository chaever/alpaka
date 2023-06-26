/* Copyright 2022 Benjamin Worpitz, Andrea Bocci, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/alpaka.hpp"

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !BOOST_LANG_CUDA
#    error If ALPAKA_ACC_GPU_CUDA_ENABLED is set, the compiler has to support CUDA!
#endif

#if defined(ALPAKA_ACC_GPU_HIP_ENABLED) && !BOOST_LANG_HIP
#    error If ALPAKA_ACC_GPU_HIP_ENABLED is set, the compiler has to support HIP!
#endif

#include "alpaka/test/Check.hpp"
#include "alpaka/test/queue/Queue.hpp"

#include <utility>

namespace alpaka::test
{
    //! The fixture for executing a kernel on a given accelerator.
    template<typename TAcc>
    class KernelExecutionFixture
    {
    public:
        using Acc = TAcc;
        using Dim = alpaka::Dim<Acc>;
        using Idx = alpaka::Idx<Acc>;
        using PlatformAcc = Pltf<Acc>;
        using DevAcc = Dev<Acc>;
        using PltfAcc = Pltf<DevAcc>;
        using QueueAcc = test::DefaultQueue<DevAcc>;
        using WorkDiv = WorkDivMembers<Dim, Idx>;

        KernelExecutionFixture(WorkDiv workDiv)
            : m_devHost(getDevByIdx(m_platformHost, 0))
            , m_devAcc(getDevByIdx(m_platformAcc, 0))
            , m_queue(m_devAcc)
            , m_workDiv(std::move(workDiv))
        {
        }

        template<typename TExtent>
        KernelExecutionFixture(TExtent const& extent)
            : KernelExecutionFixture(getValidWorkDiv<Acc>(
                getDevByIdx(m_platformAcc, 0),
                extent,
                Vec<Dim, Idx>::ones(),
                false,
                GridBlockExtentSubDivRestrictions::Unrestricted))
        {
        }

        template<typename TKernelFnObj, typename... TArgs>
        auto operator()(TKernelFnObj const& kernelFnObj, TArgs&&... args) -> bool
        {
            // Allocate the result value
            auto bufAccResult = allocBuf<bool, Idx>(m_devAcc, static_cast<Idx>(1u));
            memset(m_queue, bufAccResult, static_cast<std::uint8_t>(true));

            exec<Acc>(m_queue, m_workDiv, kernelFnObj, getPtrNative(bufAccResult), std::forward<TArgs>(args)...);

            // Copy the result value to the host
            auto bufHostResult = allocBuf<bool, Idx>(m_devHost, static_cast<Idx>(1u));
            memcpy(m_queue, bufHostResult, bufAccResult);
            wait(m_queue);

            auto const result = *getPtrNative(bufHostResult);

            return result;
        }

    private:
        PltfCpu m_platformHost;
        DevCpu m_devHost;
        PlatformAcc m_platformAcc;
        DevAcc m_devAcc;
        QueueAcc m_queue;
        WorkDiv m_workDiv;
    };
} // namespace alpaka::test
