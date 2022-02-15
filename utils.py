import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from multiprocessing import Pool
import os, time, random
from tqdm import tqdm
import shutil
import copy
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from model import VirusOnNetwork, State, number_infected_not_vac, number_infected_vac


class multi_simulation(object):
    def __init__(self, storage_path, simulation_params):
        self.storage_path = storage_path
        self.processes_num = 10* len(simulation_params)
        self.simulation_params = simulation_params

    def run_sumulation(self, iter_n, simulation_para):
        print('Run task %s (%s)...' % (simulation_para['savename'], os.getpid()))
        folder = self.storage_path + '{}/'.format(simulation_para['savename'])
        # print(folder)
        if not os.path.exists(folder):
            # print(folder)
            os.makedirs(folder)
            print(folder)
        start = time.time()
        model = VirusOnNetwork(p=simulation_para['vac_capacity'] * 1 / 365,
                               seeds={'graph': 1997, 'infect_id': 1000, 'dist_1': 1, 'dist_2': 2, 'dist_3': 3},
                               area_ratio={'city': simulation_para['city'], 'rural': simulation_para['rural']},
                               type_ratio={'A65': simulation_para['A65'], 'B65': simulation_para['B65']},
                               resis_size=simulation_para['resis_size'],
                               num_nodes=simulation_para['num_nodes'],
                               avg_node_degree=simulation_para['avg_node_degree'],
                               initial_outbreak_size=simulation_para['initial_outbreak_size'],
                               death_chance=simulation_para['death_chance'],
                               recovery_chance=simulation_para['recovery_chance'],
                               vac_effective=simulation_para['vac_effective'],
                               intention=simulation_para['intention'], Ri=simulation_para['Ri'],
                               eff_day=simulation_para['eff_day'],
                               full_vacc_size=simulation_para['full_vacc_size'],
                               defect_vac=simulation_para['defect_vac'])
        model.run_model(simulation_para['step'])
        result_i = model.datacollector.get_model_vars_dataframe()
        result_i.insert(result_i.shape[1], 'Path_No', iter_n)
        result_i.to_csv(folder + '{}.csv'.format(iter_n))
        end = time.time()
        print('Task %s runs %0.2f seconds.' % (simulation_para['savename'], (end - start)))

    def multi_save(self):
        print('Parent process %s.' % os.getpid())
        pool = Pool(processes=self.processes_num)
        n = self.processes_num
        start = time.time()
        for i in tqdm(range(self.processes_num)):
            pool.apply_async(self.run_sumulation, ((i + 1) % 10, self.simulation_params[i // 10]))
        print('Waiting for all subprocesses done...')
        pool.close()
        pool.join()
        end = time.time()
        print('All subprocesses done.')
        print('Task runs %0.2f seconds.' % (end - start))

    def merge(self):
        for x in self.simulation_params:
            result = pd.DataFrame(list())
            for i in os.listdir(self.storage_path + x['savename']):
                result_i = pd.read_csv(self.storage_path + x['savename'] + '/' + i)
                result = result.append(result_i)
            result.to_csv(self.storage_path + 'finals/' + '{}.csv'.format(x['savename']),index=False)
            shutil.rmtree(self.storage_path + x['savename'])
