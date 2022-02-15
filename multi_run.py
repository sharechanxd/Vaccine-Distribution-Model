import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from multiprocessing import Pool
import os, time, random
from tqdm import tqdm
import copy
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from model import VirusOnNetwork, State, number_infected_not_vac, number_infected_vac
from utils import multi_simulation


def prepare():
    excel_path = 'VacModel_input_batch.xlsx'
    input_para = pd.read_excel(excel_path)
    input_para = input_para.to_dict(orient='records')
    print(
        "Please input your choice for scenarios. For specific scenarios, type like '2 3 4 12', if you want to run all "
        "the scenarios, just type 'all' ")
    choices = 'all'
    if choices == 'all':
        scenario_paras = input_para
    else:
        scenario_paras = [input_para[int(i) - 1] for i in choices.split(' ')]
    return scenario_paras


if __name__ == '__main__':
    scenario_paras = prepare()
    # print(scenario_paras)
    sims = multi_simulation(storage_path='C:/Users/Administrator/Documents/sims/', simulation_params=scenario_paras)
    sims.multi_save()
    sims.merge()
