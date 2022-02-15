import pandas as pd

excel_path = 'VacModel_input_batch.xlsx'
input_para = pd.read_excel(excel_path)
# print(input_para.dtypes)
input_para = input_para.to_dict(orient='records')
# print(input_para)
simulation_para = input_para[0]
print(simulation_para)
from model import VirusOnNetwork, State, number_infected_not_vac, number_infected_vac

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
result_i.insert(result_i.shape[1], 'Path_No', 3)
result_i.to_csv('{}.csv'.format(4))
