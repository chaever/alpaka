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

            /** copy constructor
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable(Variable const&) = default;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable(Variable&&) = default;

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE Variable& operator=(Variable&&) = default;

             /** get element for the worker
             *
             * @tparam T_Idx any type which can be implicit casted to an integral type
             * @param idx index within the array
             *
             * @{
             */
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE decltype(auto) operator[](Idx const idx) const
            {
                constexpr auto laneCount = laneCount_v<pack_t>;
                return getElem(BaseArray::operator[](idx.getWorkerElemIdx()/laneCount), idx.getWorkerElemIdx()%laneCount);
            }

            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE decltype(auto) operator[](Idx const idx)
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
            using result_elem_t = decltype(std::declval<T_Type_Left>() op std::declval<T_Type_Right>());\
            using left_t = std::decay_t<decltype(left)>;\
            using right_t = std::decay_t<decltype(right)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<typename left_t::pack_t>, std::decay_t<typename right_t::pack_t>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            /*make sure that elemCount of packs matches*/\
            static_assert(laneCount_v<typename left_t::pack_t> == laneCount_v<typename right_t::pack_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<left_t::numSimdPacks; ++i){\
                tmp.packAt(i) = convertPack<typename decltype(tmp)::pack_t>(left.packAt(i) op right.packAt(i));\
            }\
            return tmp;\
        }\
        /*var op scalar*/\
        template<typename T_Type_Left, typename T_SizeInd_Left, typename T_Config, typename T_Type_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Type_Left, T_Config, T_SizeInd_Left> const left, T_Type_Right const right){\
            static_assert(std::is_arithmetic_v<std::decay_t<T_Type_Right>>);\
            using result_elem_t = decltype(std::declval<T_Type_Left>() op std::declval<T_Type_Right>());\
            using left_t = std::decay_t<decltype(left)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<typename left_t::pack_t>, std::decay_t<T_Type_Right>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<left_t::numSimdPacks; ++i){\
                tmp.packAt(i) = convertPack<typename decltype(tmp)::pack_t>(left.packAt(i) op right);\
            }\
            return tmp;\
        }\
        /*scalar op var*/\
        template<typename T_Type_Left, typename T_Type_Right, typename T_Config, typename T_SizeInd_Right>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (T_Type_Left const left, Variable<T_Type_Right, T_Config, T_SizeInd_Right> const right){\
            static_assert(std::is_arithmetic_v<std::decay_t<T_Type_Left>>);\
            using result_elem_t = decltype(std::declval<T_Type_Left>() op std::declval<T_Type_Right>());\
            using right_t = std::decay_t<decltype(right)>;\
            using size_indicator_t = alpaka::lockstep::packOperatorSizeInd_t<std::decay_t<T_Type_Left>, std::decay_t<typename right_t::pack_t>, result_elem_t>;\
            static_assert(!std::is_same_v<bool, size_indicator_t>);\
            Variable<result_elem_t, T_Config, size_indicator_t> tmp;\
            for(auto i=0u; i<right_t::numSimdPacks; ++i){\
                tmp.packAt(i) = convertPack<typename decltype(tmp)::pack_t>(left op right.packAt(i));\
            }\
            return tmp;\
        }

#define OPERATOR_DEF_VAR_PREFIX(op)\
        template<typename T_Type, typename T_Config, typename T_SizeInd>\
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr auto OPERATOR()op (Variable<T_Type, T_Config, T_SizeInd> var){\
            /*we assume that all unary operations eturn the same type, make sure that this assumption is correct*/\
            static_assert(std::is_same_v<T_Type, decltype(op std::declval<T_Type>())>);\
            for(auto i=0u; i<Variable<T_Type, T_Config, T_SizeInd>::numSimdPacks; ++i){\
                var.packAt(i) = op var.packAt(i);\
            }\
            return var;\
        }

        OPERATOR_DEF_VAR_BINARY(+)
        OPERATOR_DEF_VAR_BINARY(-)
        OPERATOR_DEF_VAR_BINARY(*)
        OPERATOR_DEF_VAR_BINARY(/)
        OPERATOR_DEF_VAR_BINARY(%)
        OPERATOR_DEF_VAR_BINARY(&)
        OPERATOR_DEF_VAR_BINARY(&&)
        OPERATOR_DEF_VAR_BINARY(|)
        OPERATOR_DEF_VAR_BINARY(||)
        OPERATOR_DEF_VAR_BINARY(<)
        OPERATOR_DEF_VAR_BINARY(>)
        OPERATOR_DEF_VAR_BINARY(<<)
        OPERATOR_DEF_VAR_BINARY(>>)

        OPERATOR_DEF_VAR_PREFIX(!)
        OPERATOR_DEF_VAR_PREFIX(~)

        template<typename T_Type, typename T_Config, typename T_SizeIndDest, typename T_SizeIndMask>
        struct MaskedAssignReference{
            Variable<T_Type, T_Config, T_SizeIndDest> & dest;
            Variable<bool, T_Config, T_SizeIndMask> const mask;

            template<typename T_TypeOperand, typename T_SizeIndOperand>
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void operator= (Variable<T_TypeOperand, T_Config, T_SizeIndOperand> value){
                for(auto i=0u; i<std::decay_t<decltype(dest)>::numSimdPacks; ++i){
                    conditionallyAssignable(dest.packAt(i), mask.packAt(i)) = value.packAt(i);
                }
            }

#define ASSIGN_OP(op)\
            template<typename T_TypeOperand, typename T_SizeIndOperand>\
            ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr void OPERATOR()op (Variable<T_TypeOperand, T_Config, T_SizeIndOperand> value){\
                for(auto i=0u; i<std::decay_t<decltype(dest)>::numSimdPacks; ++i){\
                    conditionallyAssignable(dest.packAt(i), mask.packAt(i)) op value.packAt(i);\
                }\
            }

            ASSIGN_OP(+=)
            ASSIGN_OP(-=)
            ASSIGN_OP(*=)
            ASSIGN_OP(/=)
            ASSIGN_OP(%=)

#undef ASSIGN_OP
        };

    } // namespace lockstep
} // namespace alpaka

        template<typename T_Type, typename T_Config, typename T_SizeInd>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) std::abs(alpaka::lockstep::Variable<T_Type, T_Config, T_SizeInd> var){
            using ret_t = decltype(std::abs(var[std::declval<alpaka::lockstep::Idx>()]));
            static_assert(std::is_same_v<T_Type, ret_t>);
            for(auto i=0u; i<alpaka::lockstep::Variable<T_Type, T_Config, T_SizeInd>::numSimdPacks; ++i){
                var.packAt(i) = std::abs(var.packAt(i));
            }
            return var;
        }

        template<typename T_Type, typename T_Config, typename T_SizeIndAssignee, typename T_SizeIndMask>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) where(alpaka::lockstep::Variable<T_Type, T_Config, T_SizeIndAssignee> & var, alpaka::lockstep::Variable<bool, T_Config, T_SizeIndMask> const& mask){
            return alpaka::lockstep::MaskedAssignReference(var, mask);
        }

namespace alpaka
{
    namespace lockstep
    {

#undef OPERATOR_DEF_VAR_ASSIGN
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

        //load contiguous elements into a ctxVar, starting at the specified pointer.
        //the number of loaded elements is the domain size.
        template<typename T_Worker, typename T_Config, typename T_Elem>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE auto makeVarFromContigousMemory(ForEach<T_Worker, T_Config> const& forEach, T_Elem const * const ptr){

            //allocate uninitialized memory
            auto tmpVar{makeVar<T_Elem>(forEach)};

            using pack_t = typename std::decay_t<decltype(tmpVar)>::pack_t;

            // number of iterations each worker can safely execute without boundary checks
            constexpr uint32_t guaranteedPackLoadsPerWorker = T_Config::domainSize / (T_Config::numWorkers * laneCount_v<pack_t>);
            if constexpr(guaranteedPackLoadsPerWorker>0){
                for(auto i=0u; i<guaranteedPackLoadsPerWorker; ++i){
                    tmpVar.packAt(i) = alpaka::lockstep::loadPackUnaligned<pack_t>(ptr+i*laneCount_v<pack_t>);
                }
            }

            constexpr uint32_t elementLoadsLeftForAllWorkers = T_Config::domainSize - guaranteedPackLoadsPerWorker * T_Config::numWorkers * laneCount_v<pack_t>;
            constexpr bool incompletePackRequired = elementLoadsLeftForAllWorkers % laneCount_v<pack_t> != 0u;
            constexpr uint32_t singleElemLoadsRequired = T_Config::domainSize % laneCount_v<pack_t> != 0u;
            constexpr uint32_t guaranteedPackLoadsTotal = guaranteedPackLoadsPerWorker * T_Config::numWorkers;
            constexpr uint32_t workersWithLoadsLeft = alpaka::core::divCeil(elementLoadsLeftForAllWorkers, laneCount_v<pack_t>);

            if constexpr(elementLoadsLeftForAllWorkers > 0){

                const uint32_t index = guaranteedPackLoadsTotal + forEach.getWorkerIdx();

                if constexpr(incompletePackRequired){
                    const bool hasCompletePackLeft = forEach.getWorkerIdx() < workersWithLoadsLeft - 1;
                    const bool hasIncompletePackLeft = forEach.getWorkerIdx() ==  workersWithLoadsLeft - 1;
                    if(hasCompletePackLeft){
                        tmpVar.packAt(index) = alpaka::lockstep::loadPackUnaligned<pack_t>(ptr+index*laneCount_v<pack_t>);
                    }
                    if(hasIncompletePackLeft){
                        for(auto i=0u; i<singleElemLoadsRequired; ++i){
                            tmpVar[index*laneCount_v<pack_t>+i] = ptr[index*laneCount_v<pack_t>+i];
                        }
                    }
                }else{
                    const bool currentWorkerHasOneMorePack = forEach.getWorkerIdx() < workersWithLoadsLeft;
                    if(currentWorkerHasOneMorePack){
                        tmpVar.packAt(index) = alpaka::lockstep::loadPackUnaligned<pack_t>(ptr+index*laneCount_v<pack_t>);
                    }
                }
            }
            return tmpVar;
        }

        template<typename T_Worker, typename T_Elem, typename T_Config, typename T_SizeInd>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void storeVarToContigousMemory(ForEach<T_Worker, T_Config> const& forEach, Variable<T_Elem, T_Config, T_SizeInd> const& var, T_Elem * const ptr){

            using pack_t = typename std::decay_t<decltype(var)>::pack_t;

            // number of iterations each worker can safely execute without boundary checks
            constexpr uint32_t guaranteedPackStoresPerWorker = T_Config::domainSize / (T_Config::numWorkers * laneCount_v<pack_t>);
            if constexpr(guaranteedPackStoresPerWorker>0){
                for(auto i=0u; i<guaranteedPackStoresPerWorker; ++i){
                    alpaka::lockstep::storePackUnaligned<pack_t>(var.packAt(i), ptr+i*laneCount_v<pack_t>);
                }
            }

            constexpr uint32_t elementStoresLeftForAllWorkers = T_Config::domainSize - guaranteedPackStoresPerWorker * T_Config::numWorkers * laneCount_v<pack_t>;
            constexpr bool incompletePackRequired = elementStoresLeftForAllWorkers % laneCount_v<pack_t> != 0u;
            constexpr uint32_t singleElemStoresRequired = T_Config::domainSize % laneCount_v<pack_t> != 0u;
            constexpr uint32_t guaranteedPackStoresTotal = guaranteedPackStoresPerWorker * T_Config::numWorkers;
            constexpr uint32_t workersWithStoresLeft = alpaka::core::divCeil(elementStoresLeftForAllWorkers, laneCount_v<pack_t>);

            if constexpr(elementStoresLeftForAllWorkers > 0){

                const uint32_t index = guaranteedPackStoresTotal + forEach.getWorkerIdx();

                if constexpr(incompletePackRequired){
                    const bool hasCompletePackLeft = forEach.getWorkerIdx() < workersWithStoresLeft - 1;
                    const bool hasIncompletePackLeft = forEach.getWorkerIdx() ==  workersWithStoresLeft - 1;
                    if(hasCompletePackLeft){
                        alpaka::lockstep::storePackUnaligned<pack_t>(var.packAt(index), ptr+index*laneCount_v<pack_t>);
                    }
                    if(hasIncompletePackLeft){
                        for(auto i=0u; i<singleElemStoresRequired; ++i){
                            ptr[index*laneCount_v<pack_t>+i] = var[index*laneCount_v<pack_t>+i];
                        }
                    }
                }else{
                    const bool currentWorkerHasOneMorePack = forEach.getWorkerIdx() < workersWithStoresLeft;
                    if(currentWorkerHasOneMorePack){
                        alpaka::lockstep::storePackUnaligned<pack_t>(var.packAt(index), ptr+index*laneCount_v<pack_t>);
                    }
                }
            }
        }

        template<typename T_Elem, typename T_Config, typename T_SizeInd>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) getElem(Variable<T_Elem, T_Config, T_SizeInd> const& var, uint32_t const i){
            constexpr auto laneCount = laneCount_v<typename std::decay_t<decltype(var)>::pack_t>;
            return getElem(var.packAt(i/laneCount), i%laneCount);
        }

        template<typename T_Elem, typename T_Config, typename T_SizeInd>
        ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE constexpr decltype(auto) getElem(Variable<T_Elem, T_Config, T_SizeInd> & var, uint32_t const i){
            constexpr auto laneCount = laneCount_v<typename std::decay_t<decltype(var)>::pack_t>;
            return getElem(var.packAt(i/laneCount), i%laneCount);
        }

        template<typename T_Elem, typename T_Config, typename T_SizeInd>
        constexpr auto laneCount_v<Variable<T_Elem, T_Config, T_SizeInd>> = T_Config::domainSize;
    } // namespace lockstep
} // namespace alpaka
