# -*- coding: utf-8 -*-
import nest
import numpy as np
import os
from datetime import datetime
import yaml
from dicthash import dicthash

from helpers.lognormal import mu_sigma_lognorm


class Simulation():
    """
    Handles the setup of the network parameters and
    provides functions to connect the network and devices.

    Parameters
    ----------
    sim_dict : dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: default_sim_params.py)
    net_dict : dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)
    """

    def __init__(self, sim_dict, net_dict):
        self.sim_dict = sim_dict
        self.net_dict = net_dict

    def set_data_path(self, data_path):
        """
        Sets the path for the output files.

        Parameters
        ----------
        data_path : string
        """
        self.data_path = data_path
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            self.spike_path = os.path.join(self.data_path, 'spikes')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            self.volt_path = os.path.join(self.data_path, 'voltages')

        if nest.Rank() == 0:
            if os.path.isdir(data_path):
                print('data directory already exists')
            else:
                os.mkdir(data_path)
                print('data directory created')
            print('Data will be written to %s' % self.data_path)

            if 'spike_recorder' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.spike_path):
                    os.mkdir(self.spike_path)
            if 'voltmeter' in self.sim_dict['rec_dev']:
                if not os.path.isdir(self.volt_path):
                    os.mkdir(self.volt_path)

    def setup_nest(self, num_threads):
        """
        Hands parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        Parameters
        ----------
        num_threads : int
            Local number of threads (per MPI process).
        """
        nest.ResetKernel()
        master_seed = self.sim_dict['master_seed']
        if nest.Rank() == 0:
            print('Master seed: %i ' % master_seed)
        nest.SetKernelStatus(
            {'local_num_threads': num_threads}
            )
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        if nest.Rank() == 0:
            print('Number of total processes: %i' % N_tp)

        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'rng_seed': master_seed,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time'],
            }
        nest.SetKernelStatus(kernel_dict)

    def create_populations(self):
        """
        Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from a
        normal distribution.
        """
        # Create cortical populations.
        print('Memory on rank {} before creating populations: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))
        self.pops = {}
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        local_num_threads = nest.GetKernelStatus('local_num_threads')
        with open(pop_file_name, 'w+') as pop_file:
            for pop, nn in self.net_dict['neuron_numbers'].items():
                if nn > 0:
                    if pop[-1] == 'E':
                        neuron_model_pop = self.net_dict['neuron_model_E']
                        neuron_params_pop = self.net_dict['neuron_params_E']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                    elif pop[-1] == 'I':
                        neuron_model_pop = self.net_dict['neuron_model_I']
                        neuron_params_pop = self.net_dict['neuron_params_I']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']
                    else:
                        raise NotImplementedError(
                            "Populations have to be E or I"
                        )
                    population = nest.Create(neuron_model_pop, nn)
                    population.set(neuron_params_pop)

                    # Assign DC input
                    nest.SetStatus(
                        population, {
                            'I_e': self.net_dict['dc_drive'].loc[pop]
                        }
                    )

                    # population.set(neuron_params_pop)
                    population.V_m = nest.random.normal(
                        mean=self.sim_dict['V0_mean'],
                        std=self.sim_dict['V0_sd']
                    )
                    # Neuron parameters
                    for prm, dist_dict in nrn_prm_dist_pop.items():
                        if dist_dict['rel_sd'] > 0:
                            param_dist = dist_dict['distribution']
                            if param_dist == 'lognormal':
                                mean_prm = neuron_params_pop[prm]
                                offset_prm = 0.
                                if prm == 'V_th':
                                    mean_prm -= neuron_params_pop['E_L']
                                    offset_prm += neuron_params_pop['E_L']
                                assert mean_prm > 0
                                mu_param, sigma_param = mu_sigma_lognorm(
                                    mean=mean_prm,
                                    rel_sd=dist_dict['rel_sd']
                                )
                                population.set({prm:
                                    offset_prm + nest.random.lognormal(
                                    mean=mu_param,
                                    std=sigma_param
                                )})
                            else:
                                err_msg = "Parameter distribution "
                                err_msg += f"{param_dist} not implemented."
                                raise NotImplementedError(err_msg)

                    self.pops[pop] = population
                    pop_file.write('{};{};{}\n'.format(
                        pop, population[0].get()['global_id'], population[-1].get()['global_id']
                    ))
        print('Memory on rank {} after creating populations: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))

    # def create_populations_Abeta(self):
    #     """
    #     创建神经元群体，包含受Aβ影响的神经元 (Creates neuronal populations with Aβ-affected neurons)

    #     此函数负责创建大脑仿真模型中的所有神经元群体，包括正常神经元和受β-淀粉样蛋白(Aβ)
    #     影响的神经元。这对于阿尔茨海默病的病理建模至关重要。
        
    #     主要功能:
    #     - 根据Abeta_ratio创建受Aβ影响的神经元和正常神经元
    #     - 分别设置两类神经元的不同参数
    #     - 记录Aβ影响神经元的GID到专门文件
    #     - 监控内存使用情况
    #     - 记录所有群体信息
    #     """
        
    #     # ==================== 内存监控和初始化 ====================
    #     print('Memory on rank {} before creating populations: {:.2f}MB'.format(
    #         nest.Rank(), self._getMemoryMB()
    #     ))
        
    #     # 初始化群体字典
    #     self.pops = {}
        
    #     # ==================== 文件设置 ====================
    #     # 设置群体ID记录文件路径
    #     pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
    #     # 设置Aβ影响神经元ID记录文件路径
    #     abeta_file_name = os.path.join(self.data_path, 'Abeta_GID.txt')
        
    #     # 获取本地线程数
    #     local_num_threads = nest.GetKernelStatus('local_num_threads')
        
    #     # ==================== 双文件写入主循环 ====================
    #     # 同时打开两个文件：群体信息文件和Aβ神经元文件
    #     with open(pop_file_name, 'w+') as pop_file, open(abeta_file_name, 'w+') as abeta_file:
            
    #         # 写入Aβ文件头部信息
    #         abeta_file.write('# Abeta-affected neurons Global IDs\n')
    #         abeta_file.write('# Format: Population_Name;Start_GID;End_GID;Neuron_Count\n')
            
    #         # 遍历所有神经元群体
    #         for pop, nn in self.net_dict['neuron_numbers'].items():
    #             if nn > 0:  # 只处理有神经元的群体
                    
    #                 # ==================== 神经元类型识别和参数加载 ====================
    #                 if pop[-1] == 'E':  # 兴奋性神经元
    #                     # 正常兴奋性神经元参数
    #                     neuron_model_pop = self.net_dict['neuron_model_E']
    #                     neuron_params_pop = self.net_dict['neuron_params_E']
    #                     nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                        
    #                     # Aβ影响的兴奋性神经元参数
    #                     Abeta_ratio = self.net_dict["Abeta_ratio_E"]
    #                     neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_E"]
    #                     neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_E']
    #                     nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_E']
                        
    #                 elif pop[-1] == 'I':  # 抑制性神经元
    #                     # 正常抑制性神经元参数
    #                     neuron_model_pop = self.net_dict['neuron_model_I']
    #                     neuron_params_pop = self.net_dict['neuron_params_I']
    #                     nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']
                        
    #                     # Aβ影响的抑制性神经元参数
    #                     Abeta_ratio = self.net_dict["Abeta_ratio_I"]
    #                     neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_I"]
    #                     neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_I']
    #                     nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_I']
                        
    #                 else:
    #                     raise NotImplementedError("Populations have to be E or I")
                    
    #                 # ==================== 神经元数量计算 ====================
    #                 # 根据Aβ比例计算受影响和正常神经元的数量
    #                 nn_Abeta = int(np.round(nn * Abeta_ratio))  # 受Aβ影响的神经元数量
    #                 nn_normal = int(nn - nn_Abeta)             # 正常神经元数量
                    
    #                 print(f'Population {pop}: Total={nn}, Abeta-affected={nn_Abeta}, Normal={nn_normal}, Abeta_ratio={Abeta_ratio:.3f}')
                    
    #                 # ==================== 分别创建两类神经元群体 ====================
    #                 created_populations = []  # 存储创建的子群体
                    
    #                 # 创建受Aβ影响的神经元（如果数量大于0）
    #                 if nn_Abeta > 0:
    #                     pop_Abeta = nest.Create(neuron_model_pop_Abeta, nn_Abeta)
    #                     pop_Abeta.set(neuron_params_pop_Abeta)
    #                     created_populations.append(pop_Abeta)
                        
    #                     # 记录Aβ影响神经元的GID信息
    #                     abeta_file.write('{};{};{};{}\n'.format(
    #                         pop,                                    # 群体名称
    #                         pop_Abeta[0].get()['global_id'],       # 起始GID
    #                         pop_Abeta[-1].get()['global_id'],      # 结束GID
    #                         nn_Abeta                               # 神经元数量
    #                     ))
                        
    #                     print(f'  Created {nn_Abeta} Abeta-affected neurons (GID: {pop_Abeta[0].get()["global_id"]}-{pop_Abeta[-1].get()["global_id"]})')
                    
    #                 # 创建正常神经元（如果数量大于0）
    #                 if nn_normal > 0:
    #                     pop_normal = nest.Create(neuron_model_pop, nn_normal)
    #                     pop_normal.set(neuron_params_pop)
    #                     created_populations.append(pop_normal)
                        
    #                     print(f'  Created {nn_normal} normal neurons (GID: {pop_normal[0].get()["global_id"]}-{pop_normal[-1].get()["global_id"]})')
                    
    #                 # ==================== 合并群体 ====================
    #                 # 将Aβ影响和正常神经元合并为一个群体
    #                 if created_populations:
    #                     if len(created_populations) == 2:
    #                         population = created_populations[0] + created_populations[1]  # Aβ + 正常
    #                     else:
    #                         population = created_populations[0]  # 只有一种类型
    #                 else:
    #                     continue  # 如果没有创建任何神经元，跳过此群体
                    
    #                 # ==================== 直流驱动设置 ====================
    #                 # 为整个群体（包括Aβ和正常神经元）分配相同的直流输入
    #                 nest.SetStatus(
    #                     population, {
    #                         'I_e': self.net_dict['dc_drive'].loc[pop]
    #                     }
    #                 )
                    
    #                 # ==================== 膜电位初始化 ====================
    #                 # 所有神经元使用相同的初始膜电位分布
    #                 population.V_m = nest.random.normal(
    #                     mean=self.sim_dict['V0_mean'],
    #                     std=self.sim_dict['V0_sd']
    #                 )
                    
    #                 # ==================== 参数分布设置 ====================
    #                 # 分别为Aβ影响和正常神经元设置参数分布
                    
    #                 # 1. 为Aβ影响的神经元设置参数分布
    #                 if nn_Abeta > 0:
    #                     for prm, dist_dict in nrn_prm_dist_pop_Abeta.items():
    #                         if dist_dict['rel_sd'] > 0:
    #                             param_dist = dist_dict['distribution']
    #                             if param_dist == 'lognormal':
    #                                 mean_prm = neuron_params_pop_Abeta[prm]
    #                                 offset_prm = 0.
    #                                 if prm == 'V_th':
    #                                     mean_prm -= neuron_params_pop_Abeta['E_L']
    #                                     offset_prm += neuron_params_pop_Abeta['E_L']
    #                                 assert mean_prm > 0
    #                                 mu_param, sigma_param = mu_sigma_lognorm(
    #                                     mean=mean_prm,
    #                                     rel_sd=dist_dict['rel_sd']
    #                                 )
    #                                 # 只对Aβ影响的神经元应用参数
    #                                 pop_Abeta.set({prm:
    #                                     offset_prm + nest.random.lognormal(
    #                                     mean=mu_param,
    #                                     std=sigma_param
    #                                 )})
    #                             else:
    #                                 err_msg = f"Parameter distribution {param_dist} not implemented for Abeta neurons."
    #                                 raise NotImplementedError(err_msg)
                    
    #                 # 2. 为正常神经元设置参数分布
    #                 if nn_normal > 0:
    #                     for prm, dist_dict in nrn_prm_dist_pop.items():
    #                         if dist_dict['rel_sd'] > 0:
    #                             param_dist = dist_dict['distribution']
    #                             if param_dist == 'lognormal':
    #                                 mean_prm = neuron_params_pop[prm]
    #                                 offset_prm = 0.
    #                                 if prm == 'V_th':
    #                                     mean_prm -= neuron_params_pop['E_L']
    #                                     offset_prm += neuron_params_pop['E_L']
    #                                 assert mean_prm > 0
    #                                 mu_param, sigma_param = mu_sigma_lognorm(
    #                                     mean=mean_prm,
    #                                     rel_sd=dist_dict['rel_sd']
    #                                 )
    #                                 # 只对正常神经元应用参数
    #                                 pop_normal.set({prm:
    #                                     offset_prm + nest.random.lognormal(
    #                                     mean=mu_param,
    #                                     std=sigma_param
    #                                 )})
    #                             else:
    #                                 err_msg = f"Parameter distribution {param_dist} not implemented for normal neurons."
    #                                 raise NotImplementedError(err_msg)
                    
    #                 # ==================== 群体存储和记录 ====================
    #                 # 将合并后的群体存储到类字典中
    #                 self.pops[pop] = population
                    
    #                 # 记录整个群体的信息到population_GIDs.dat文件
    #                 pop_file.write('{};{};{}\n'.format(
    #                     pop,                                        # 群体名称
    #                     population[0].get()['global_id'],          # 群体起始GID
    #                     population[-1].get()['global_id']          # 群体结束GID
    #                 ))
        
    #     # ==================== 最终内存监控和统计 ====================
    #     print('Memory on rank {} after creating populations: {:.2f}MB'.format(
    #         nest.Rank(), self._getMemoryMB()
    #     ))
        
    #     # 输出创建的群体统计信息
    #     total_neurons = sum(len(pop) for pop in self.pops.values())
    #     print(f'Successfully created {len(self.pops)} populations with {total_neurons} total neurons.')
    #     print(f'Population GIDs saved to: {pop_file_name}')
    #     print(f'Abeta-affected neuron GIDs saved to: {abeta_file_name}')

    # ==================== 补充函数说明 ====================
    # """
    # 此修改版本的主要改进:

    # 1. Aβ病理建模 (Aβ Pathology Modeling):
    # - 根据Abeta_ratio精确计算受影响神经元数量
    # - 分别创建和配置Aβ影响和正常神经元
    # - 使用不同的神经元模型和参数来模拟病理变化

    # 2. 数据记录增强 (Enhanced Data Recording):
    # - 创建专门的Abeta_GID.txt文件记录受影响神经元
    # - 详细记录每个群体中Aβ神经元的GID范围
    # - 添加统计信息和创建过程日志

    # 3. 参数分布优化 (Parameter Distribution Optimization):
    # - 分别为Aβ和正常神经元设置不同的参数分布
    # - 保持参数设置的生物学现实性
    # - 支持不同的病理严重程度建模

    # 4. 错误处理和调试 (Error Handling and Debugging):
    # - 添加详细的创建过程打印信息
    # - 改进错误消息的特异性
    # - 添加数量验证和统计输出

    # 5. 计算效率 (Computational Efficiency):
    # - 优化神经元创建流程
    # - 减少不必要的参数设置重复
    # - 改进内存使用监控

    # 这个版本特别适合阿尔茨海默病的病理建模研究，能够精确控制病理神经元的比例和特性。
    # """

    def create_populations_Abeta(self):
        """
        创建神经元群体，包含受Aβ影响的神经元 (Creates neuronal populations with Aβ-affected neurons)
        
        优化版本：解决文件缓冲区过大问题，适合SLURM集群大规模仿真
        """
        
        # ==================== 内存监控和初始化 ====================
        print('Memory on rank {} before creating populations: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))
        
        # 初始化群体字典和统计变量
        self.pops = {}
        processed_populations = 0
        total_abeta_neurons = 0
        
        # ==================== 文件设置 ====================
        pop_file_name = os.path.join(self.data_path, 'population_GIDs.dat')
        abeta_file_name = os.path.join(self.data_path, 'Abeta_GID.txt')
        
        # 计算总群体数量用于进度监控
        total_populations = len([pop for pop, nn in self.net_dict['neuron_numbers'].items() if nn > 0])
        
        # ==================== 优化的双文件写入主循环 ====================
        # 使用行缓冲模式，每行立即写入磁盘
        with open(pop_file_name, 'w+', buffering=1) as pop_file, \
            open(abeta_file_name, 'w+', buffering=1) as abeta_file:
            
            # 写入Aβ文件头部信息
            abeta_file.write('# Abeta-affected neurons Global IDs for Alzheimer\'s Disease Simulation\n')
            abeta_file.write('# Generated by NEST simulator on SLURM cluster\n')
            abeta_file.write('# Format: Population_Name;Start_GID;End_GID;Neuron_Count\n')
            
            # 定义刷新间隔（适合集群环境）
            FLUSH_INTERVAL = max(1, total_populations // 20)  # 每5%进度刷新一次
            MEMORY_CHECK_INTERVAL = max(1, total_populations // 10)  # 每10%检查内存
            
            # 遍历所有神经元群体
            for pop, nn in self.net_dict['neuron_numbers'].items():
                if nn > 0:  # 只处理有神经元的群体
                    
                    # ==================== 神经元类型识别和参数加载 ====================
                    if pop[-1] == 'E':  # 兴奋性神经元
                        # 正常兴奋性神经元参数
                        neuron_model_pop = self.net_dict['neuron_model_E']
                        neuron_params_pop = self.net_dict['neuron_params_E']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_E']
                        
                        # Aβ影响的兴奋性神经元参数
                        Abeta_ratio = self.net_dict["Abeta_ratio_E"]
                        neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_E"]
                        neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_E']
                        nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_E']
                        
                    elif pop[-1] == 'I':  # 抑制性神经元
                        # 正常抑制性神经元参数
                        neuron_model_pop = self.net_dict['neuron_model_I']
                        neuron_params_pop = self.net_dict['neuron_params_I']
                        nrn_prm_dist_pop = self.net_dict['neuron_param_dist_I']
                        
                        # Aβ影响的抑制性神经元参数
                        Abeta_ratio = self.net_dict["Abeta_ratio_I"]
                        neuron_model_pop_Abeta = self.net_dict["neuron_model_Abeta_I"]
                        neuron_params_pop_Abeta = self.net_dict['neuron_params_Abeta_I']
                        nrn_prm_dist_pop_Abeta = self.net_dict['neuron_param_dist_Abeta_I']
                        
                    else:
                        raise NotImplementedError("Populations have to be E or I")
                    
                    # ==================== 神经元数量计算 ====================
                    nn_Abeta = int(np.round(nn * Abeta_ratio))  # 受Aβ影响的神经元数量
                    nn_normal = int(nn - nn_Abeta)             # 正常神经元数量
                    total_abeta_neurons += nn_Abeta
                    
                    print(f'Population {pop}: Total={nn}, Abeta-affected={nn_Abeta}, Normal={nn_normal}, Abeta_ratio={Abeta_ratio:.3f}')
                    
                    # ==================== 分别创建两类神经元群体 ====================
                    created_populations = []  # 存储创建的子群体
                    
                    # 创建受Aβ影响的神经元（如果数量大于0）
                    if nn_Abeta > 0:
                        pop_Abeta = nest.Create(neuron_model_pop_Abeta, nn_Abeta)
                        pop_Abeta.set(neuron_params_pop_Abeta)
                        created_populations.append(pop_Abeta)
                        
                        # 立即记录Aβ影响神经元的GID信息（行缓冲会立即写入）
                        abeta_file.write('{};{};{};{}\n'.format(
                            pop,                                    # 群体名称
                            pop_Abeta[0].get()['global_id'],       # 起始GID
                            pop_Abeta[-1].get()['global_id'],      # 结束GID
                            nn_Abeta                               # 神经元数量
                        ))
                        
                        print(f'  Created {nn_Abeta} Abeta-affected neurons (GID: {pop_Abeta[0].get()["global_id"]}-{pop_Abeta[-1].get()["global_id"]})')
                    
                    # 创建正常神经元（如果数量大于0）
                    if nn_normal > 0:
                        pop_normal = nest.Create(neuron_model_pop, nn_normal)
                        pop_normal.set(neuron_params_pop)
                        created_populations.append(pop_normal)
                        
                        print(f'  Created {nn_normal} normal neurons (GID: {pop_normal[0].get()["global_id"]}-{pop_normal[-1].get()["global_id"]})')
                    
                    # ==================== 合并群体 ====================
                    if created_populations:
                        if len(created_populations) == 2:
                            population = created_populations[0] + created_populations[1]  # Aβ + 正常
                        else:
                            population = created_populations[0]  # 只有一种类型
                    else:
                        continue  # 如果没有创建任何神经元，跳过此群体
                    
                    # ==================== 直流驱动设置 ====================
                    nest.SetStatus(
                        population, {
                            'I_e': self.net_dict['dc_drive'].loc[pop]
                        }
                    )
                    
                    # ==================== 膜电位初始化 ====================
                    population.V_m = nest.random.normal(
                        mean=self.sim_dict['V0_mean'],
                        std=self.sim_dict['V0_sd']
                    )
                    
                    # ==================== 参数分布设置 ====================
                    # 1. 为Aβ影响的神经元设置参数分布
                    if nn_Abeta > 0:
                        for prm, dist_dict in nrn_prm_dist_pop_Abeta.items():
                            if dist_dict['rel_sd'] > 0:
                                param_dist = dist_dict['distribution']
                                if param_dist == 'lognormal':
                                    mean_prm = neuron_params_pop_Abeta[prm]
                                    offset_prm = 0.
                                    if prm == 'V_th':
                                        mean_prm -= neuron_params_pop_Abeta['E_L']
                                        offset_prm += neuron_params_pop_Abeta['E_L']
                                    assert mean_prm > 0
                                    mu_param, sigma_param = mu_sigma_lognorm(
                                        mean=mean_prm,
                                        rel_sd=dist_dict['rel_sd']
                                    )
                                    pop_Abeta.set({prm:
                                        offset_prm + nest.random.lognormal(
                                        mean=mu_param,
                                        std=sigma_param
                                    )})
                                else:
                                    err_msg = f"Parameter distribution {param_dist} not implemented for Abeta neurons."
                                    raise NotImplementedError(err_msg)
                    
                    # 2. 为正常神经元设置参数分布
                    if nn_normal > 0:
                        for prm, dist_dict in nrn_prm_dist_pop.items():
                            if dist_dict['rel_sd'] > 0:
                                param_dist = dist_dict['distribution']
                                if param_dist == 'lognormal':
                                    mean_prm = neuron_params_pop[prm]
                                    offset_prm = 0.
                                    if prm == 'V_th':
                                        mean_prm -= neuron_params_pop['E_L']
                                        offset_prm += neuron_params_pop['E_L']
                                    assert mean_prm > 0
                                    mu_param, sigma_param = mu_sigma_lognorm(
                                        mean=mean_prm,
                                        rel_sd=dist_dict['rel_sd']
                                    )
                                    pop_normal.set({prm:
                                        offset_prm + nest.random.lognormal(
                                        mean=mu_param,
                                        std=sigma_param
                                    )})
                                else:
                                    err_msg = f"Parameter distribution {param_dist} not implemented for normal neurons."
                                    raise NotImplementedError(err_msg)
                    
                    # ==================== 群体存储和记录 ====================
                    self.pops[pop] = population
                    
                    # 立即记录整个群体的信息（行缓冲会立即写入）
                    pop_file.write('{};{};{}\n'.format(
                        pop,                                        # 群体名称
                        population[0].get()['global_id'],          # 群体起始GID
                        population[-1].get()['global_id']          # 群体结束GID
                    ))
                    
                    # ==================== 进度监控和缓冲区管理 ====================
                    processed_populations += 1
                    
                    # 定期强制刷新缓冲区（适合长时间运行的SLURM作业）
                    if processed_populations % FLUSH_INTERVAL == 0:
                        pop_file.flush()
                        abeta_file.flush()
                        progress_pct = 100 * processed_populations / total_populations
                        print(f'[SLURM进度监控] 阿尔茨海默病网络构建进度: {processed_populations}/{total_populations} '
                            f'({progress_pct:.1f}%), 累积Aβ神经元: {total_abeta_neurons}')
                    
                    # 定期内存检查（防止内存溢出导致SLURM作业被杀）
                    if processed_populations % MEMORY_CHECK_INTERVAL == 0:
                        current_memory = self._getMemoryMB()
                        print(f'[内存监控] Rank {nest.Rank()}: {current_memory:.2f}MB, '
                            f'平均每个群体: {current_memory/processed_populations:.2f}MB')
                        
                        # 如果内存使用过高，发出警告
                        if current_memory > 50000:  # 50GB警告阈值
                            print(f'警告: 内存使用较高 ({current_memory:.2f}MB), 请监控SLURM作业状态')
            
            # 最终刷新确保所有数据写入磁盘
            pop_file.flush()
            abeta_file.flush()
        
        # ==================== 最终统计和验证 ====================
        final_memory = self._getMemoryMB()
        total_neurons = sum(len(pop) for pop in self.pops.values())
        
        print('=' * 80)
        print('阿尔茨海默病脑网络构建完成 (Alzheimer\'s Disease Brain Network Created)')
        print('=' * 80)
        print(f'成功创建 {len(self.pops)} 个神经元群体')
        print(f'总神经元数量: {total_neurons:,}')
        print(f'Aβ影响神经元: {total_abeta_neurons:,} ({100*total_abeta_neurons/total_neurons:.2f}%)')
        print(f'正常神经元: {total_neurons-total_abeta_neurons:,} ({100*(total_neurons-total_abeta_neurons)/total_neurons:.2f}%)')
        print(f'内存使用: Rank {nest.Rank()}: {final_memory:.2f}MB')
        print(f'群体GID文件: {pop_file_name}')
        print(f'Aβ神经元文件: {abeta_file_name}')
        print('=' * 80)
        
        # 验证文件是否正确写入
        try:
            with open(pop_file_name, 'r') as f:
                pop_lines = len(f.readlines())
            with open(abeta_file_name, 'r') as f:
                abeta_lines = len([l for l in f.readlines() if not l.startswith('#')])
            print(f'文件验证: population_GIDs.dat有{pop_lines}行, Abeta_GID.txt有{abeta_lines}行Aβ记录')
        except Exception as e:
            print(f'文件验证失败: {e}')

    def create_devices(self):
        """
        Creates the recording devices.

        Only devices which are given in sim_dict['rec_dev'] are created.
        """
        if 'spike_recorder' in self.sim_dict['rec_dev']:
            recdict = {
                'record_to': 'ascii',
                'label': os.path.join(self.spike_path, 'spike_recorder')
            }
            self.spike_recorder = nest.Create('spike_recorder', params=recdict)
        if 'voltmeter' in self.sim_dict['rec_dev']:
            recdictmem = {
                'record_to': 'ascii',
                'label': os.path.join(self.volt_path, 'voltmeter'),
                'record_from': ['V_m'],
            }
            self.voltmeter = nest.Create('multimeter', params=recdictmem)

        if 'spike_recorder' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Spike detectors created')
        if 'voltmeter' in self.sim_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Voltmeters created')

    def create_poisson(self):
        """
        Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.
        """
        if nest.Rank() == 0:
            print('Poisson background input created')
        self.poisson = {}
        for pop, nn in self.net_dict['neuron_numbers'].items():
            if nn > 0.:
                sn_ext = self.net_dict['synapses_external'].loc[pop]
                K_ext = sn_ext / nn
                rate_ext = self.net_dict['rate_ext'].loc[pop] * K_ext
                poiss = nest.Create('poisson_generator')
                poiss.set({'rate': rate_ext})
                self.poisson[pop] = poiss

    def create_single_spike(self):
        """
        Creates the single spike generator.
        """
        if nest.Rank() == 0:
            print('Single spike input created')
        self.single_spike = {}
        for pop, spike_time in self.net_dict['spike_time'].items():
            if spike_time >= 0.:
                spike = nest.Create(
                        'spike_generator',
                        params={'spike_times': [spike_time]}
                        )
                self.single_spike[pop] = spike

    def connect_neurons(self):
        """
        Connects the neuronal populations.
        """
        if nest.Rank() == 0:
            print('Connections are established')
        conn_rule = self.sim_dict['connection_rule']
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            for (area_j, layer_j, pop_j), source_pop in self.pops.items():
                synapse_nr = self.net_dict['synapses_internal'].loc[
                    (area_i, layer_i, pop_i),
                    (area_j, layer_j, pop_j)
                ]

                if synapse_nr > 0.:
                    if area_i == area_j:
                        min_delay = self.sim_dict['sim_resolution']
                        if pop_j == 'E':
                            mean_delay = self.net_dict['delay_e']
                            std_delay = self.net_dict['delay_e_sd']
                        else:
                            mean_delay = self.net_dict['delay_i']
                            std_delay = self.net_dict['delay_i_sd']
                    else:
                        min_delay = max(self.sim_dict['delay_cc_min'],
                                        self.sim_dict['sim_resolution'])
                        mean_delay = self.net_dict['delay_cc'].loc[
                            area_i, area_j
                        ]
                        std_delay = self.net_dict['delay_cc_sd'].loc[
                            area_i, area_j
                        ]
                    delay_distr = self.net_dict['delay_distribution']
                    if delay_distr == 'normal_clipped':
                        mu_delay = mean_delay
                        sigma_delay = std_delay
                    elif delay_distr == 'lognormal_clipped':
                        mu_delay, sigma_delay = mu_sigma_lognorm(
                            mean=mean_delay, rel_sd=std_delay/mean_delay
                        )
                    else:
                        err_msg = f"Delay distribution {delay_distr}"
                        err_msg += " not implemented."
                        raise NotImplementedError(err_msg)
                    weight = self.net_dict['weights'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    w_sd = self.net_dict['weights_sd'].loc[
                        (area_i, layer_i, pop_i),
                        (area_j, layer_j, pop_j)
                    ]
                    if conn_rule == 'fixed_total_number':
                        conn_dict_rec = {
                            'rule': conn_rule, 'N': synapse_nr
                        }
                    elif conn_rule == 'fixed_indegree':
                        neuron_nr = self.net_dict['neuron_numbers'].loc[
                            (area_i, layer_i, pop_i)
                        ]
                        indegree = int(np.round(synapse_nr / neuron_nr))
                        conn_dict_rec = {
                            'rule': conn_rule, 'indegree': indegree
                        }
                    else:
                        print(f'Unknown connection rule {conn_rule}.')
                        raise NotImplementedError()
                    if np.isclose(self.net_dict['p_transmit'], 1):
                        syn_dict = {'synapse_model': 'static_synapse'}
                    else:
                        syn_dict = {'synapse_model': 'bernoulli_synapse',
                                    'p_transmit': self.net_dict['p_transmit']}
                        
                    if delay_distr == 'normal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.normal(mean=mu_delay, std=sigma_delay), min_delay)
                    elif delay_distr == 'lognormal_clipped':
                        syn_dict['delay'] = nest.math.max(nest.random.lognormal(mean=mu_delay, std=sigma_delay), min_delay)

                    # Skipping connection with weight == 0.0
                    if weight == 0.0:
                        print('Found weight == 0.0 between {} and {}. Skipping connection.'.format(
                            (area_i, layer_i, pop_i),
                            (area_j, layer_j, pop_j)
                            )
                        )
                        continue
                    
                    if weight < 0:
                        syn_dict['weight'] = nest.math.min(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    else:
                        syn_dict['weight'] = nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0)
                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict
                    )
            print(
                'Connected all of area {}, layer {} and population {} '
                'on rank {}. Memory: {:.2f} MB.'.format(
                    area_i, layer_i, pop_i,
                    nest.Rank(), self._getMemoryMB()
                )
            )
        print('Memory on rank {} after creating connections: {:.2f}MB'.format(
            nest.Rank(), self._getMemoryMB()
        ))

    def connect_poisson(self):
        """
        Connects the Poisson generators to the microcircuit.
        """
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            conn_dict_poisson = {'rule': 'all_to_all'}
            weight = self.net_dict['weights_ext'].loc[
                (area_i, layer_i, pop_i)
            ]
            w_sd = self.net_dict['weights_ext_sd'].loc[
                (area_i, layer_i, pop_i)
            ]
            syn_dict_poisson = {
                'synapse_model': 'static_synapse',
                'weight': nest.math.max(nest.random.normal(mean=weight, std=w_sd), 0.0),
                'delay': self.sim_dict['sim_resolution']
            }
            nest.Connect(
                self.poisson[(area_i, layer_i, pop_i)], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
            )

    def connect_single_spike(self):
        """
        Connects a spike generator emitting a single spike.
        """
        if nest.Rank() == 0:
            print('Single spike generator input is connected')
        # Loop over all existing
        for pop, spike in self.single_spike.items():
            # Calculate indegree K for scaling weight
            nn = self.net_dict['neuron_numbers'].loc[pop]
            sn_ext = self.net_dict['synapses_external'].loc[pop]
            K_ext = sn_ext / nn
            # Choose first item in list of target gids
            target_pop = [self.pops[pop][0]]
            weight = self.net_dict['weights_ext'].loc[pop]
            # Scale weight
            weight *= 1e3*K_ext
            syn_dict_single_spike = {
                'synapse_model': 'static_synapse',
                'weight': weight,
                }
            nest.Connect(
                spike,
                target_pop,
                syn_spec=syn_dict_single_spike
            )

    def connect_devices(self):
        """ Connects the recording devices to the microcircuit."""
        if nest.Rank() == 0:
            if ('spike_recorder' in self.sim_dict['rec_dev'] and
                    'voltmeter' not in self.sim_dict['rec_dev']):
                print('Spike detector connected')
            elif ('spike_recorder' not in self.sim_dict['rec_dev'] and
                    'voltmeter' in self.sim_dict['rec_dev']):
                print('Voltmeter connected')
            elif ('spike_recorder' in self.sim_dict['rec_dev'] and
                    'voltmeter' in self.sim_dict['rec_dev']):
                print('Spike detector and voltmeter connected')
            else:
                print('no recording devices connected')
        for (area_i, layer_i, pop_i), target_pop in self.pops.items():
            if 'voltmeter' in self.sim_dict['rec_dev']:
                nest.Connect(self.voltmeter, target_pop)
            if 'spike_recorder' in self.sim_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_recorder)

    def setup(self, data_path, num_threads):
        """ Execute subfunctions of the network.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        Parameters
        ----------
        data_path : string
        num_threads : int
        """
        self.set_data_path(data_path)
        self.setup_nest(num_threads)
        
        if self.net_dict['Abeta']:
            self.create_populations_Abeta()
        else:
            self.create_populations()
        
        self.create_devices()
        self.create_poisson()
        self.create_single_spike()
        self.connect_neurons()
        self.connect_poisson()
        self.connect_single_spike()
        self.connect_devices()

    def simulate(self):
        """ Simulates the model."""
        print("{} Start simulating".format(datetime.now()))
        nest.Simulate(self.sim_dict['t_sim'])
        print("{} Simulation finished".format(datetime.now()))

    def getHash(self):
        """
        Creates a hash from simulation parameters.

        Returns
        -------
        hash : str
            Hash for the simulation
        """
        hash = dicthash.generate_hash_from_dict(self.sim_dict)
        return hash

    def _getMemoryMB(self):
        """
        Return the currently occupied memory for the job in MB.

        Returns
        -------
        currMem : float
            Curently occupied memory in MB
        """
        try:
            currMem = nest.ll_api.sli_func('memory_thisjob')/1024.
        except AttributeError:
            currMem = nest.sli_func('memory_thisjob')/1024.
        return currMem

    def dump(self, base_folder):
        """
        Exports the full simulation specification. Creates a subdirectory of
        base_folder from the simulation hash where it puts all files.

        Parameters
        ----------
        base_folder : string
            Path to base output folder
        """
        hash = self.getHash()
        out_folder = os.path.join(base_folder, hash)
        try:
            os.mkdir(out_folder)
        except OSError:
            pass

        # output simple data as yaml
        fn = os.path.join(out_folder, 'sim.yaml')
        with open(fn, 'w') as outfile:
            yaml.dump(self.sim_dict, outfile, default_flow_style=False)

def simulationDictFromDump(dump_folder):
    """
    Creates a simulation dict from the files created by Simulation.dump().

    Parameters
    ----------
    dump_folder : string`
        Folder with dumped files

    Returns
    -------
    sim_dict : dict
        Full simulation dictionary
    """
    # Read sim.yaml
    fn = os.path.join(dump_folder, 'sim.yaml')
    with open(fn, 'r') as sim_file:
        sim_dict = yaml.load(sim_file, Loader=yaml.Loader)
    return sim_dict
