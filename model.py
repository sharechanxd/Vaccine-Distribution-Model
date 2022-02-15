import math
from enum import Enum
import networkx as nx
import numpy as np
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid
from mesa.time import RandomActivation
import random
from collections import Counter
from tqdm import tqdm
from sympy import *


# 我们先用NY的数据进行仿真

class State(Enum):
    SUS = 0
    VACC_no = 1
    VACC_yes = 2
    NOT_VACCINATED_INFECTED = 3
    VACCINATED_NOT_EFFECTIVE_INFECTED = 4
    VACCINATED_EFFECTIVE_RESISTANT = 5
    INFECTED_RECOVERED_RESISTANT = 6
    DEATH = 7
    HOSP = 8


def number_accumulate_infected(model):
    return sum([1 for a in model.grid.get_all_cell_contents() if
                a.state in [State.NOT_VACCINATED_INFECTED, State.INFECTED_RECOVERED_RESISTANT,
                            State.INFECTED_RECOVERED_RESISTANT, State.DEATH, State.VACCINATED_NOT_EFFECTIVE_INFECTED]])


def number_attr(model, state, attr=['city', 'A65']):
    return sum([1 for a in model.grid.get_all_cell_contents() if
                a.state is state and a.area == attr[0] and a.sus_type == attr[1]])


def number_state(model, state):
    return sum([1 for a in model.grid.get_all_cell_contents() if a.state is state])


def number_vac(model):
    return sum([1 for a in model.grid.get_all_cell_contents() if
                a.vacc_state == 1])


def number_vac_state(model, attr=['city', 'A65']):
    return sum([1 for a in model.grid.get_all_cell_contents() if
                a.vacc_state == 1 and a.area == attr[0] and a.sus_type == attr[1]])


def number_intention_initial(model, x, attr=['city', 'A65']):
    return sum([1 for a in model.grid.get_all_cell_contents() if
                a.intention_to_vacc == x and a.area == attr[0] and a.sus_type == attr[1]])


def number_vac_no(model):
    return number_state(model, State.VACC_no)


def number_vac_yes(model):
    return number_state(model, State.VACC_yes)


def number_infected_not_vac(model):
    return number_state(model, State.NOT_VACCINATED_INFECTED)


def number_infected_vac(model):
    return number_state(model, State.VACCINATED_NOT_EFFECTIVE_INFECTED)


def number_city_susceptible_above65(model):
    return number_attr(model, State.SUS, ['city', 'A65'])


def number_city_susceptible_below65(model):
    return number_attr(model, State.SUS, ['city', 'B65'])


def number_rural_susceptible_above65(model):
    return number_attr(model, State.SUS, ['rural', 'A65'])


def number_rural_susceptible_below65(model):
    return number_attr(model, State.SUS, ['rural', 'B65'])


def number_resistant_vac(model):
    return number_state(model, State.VACCINATED_EFFECTIVE_RESISTANT)


def number_resistant_nature(model):
    return number_state(model, State.INFECTED_RECOVERED_RESISTANT)


def number_death(model):
    return number_state(model, State.DEATH)


def number_hospital(model):
    return number_state(model, State.HOSP)


class VirusOnNetwork(Model):
    """A virus model with some number of agents"""

    def __init__(
            self,
            p,
            seeds={'graph': 1997, 'infect_id': 1000, 'dist_1': 1, 'dist_2': 2, 'dist_3': 3},
            num_nodes=19453561,
            avg_node_degree=30,
            initial_outbreak_size=163580,  # Current data for the state
            death_chance=0.05399,
            recovery_chance=0.94601,
            gain_resistance_chance_after_infected=1,
            vac_effective=0.8,
            hosp_rate=0.2,
            area_ratio={'city': 0.827, 'rural': 0.173},
            type_ratio={'A65': 0.152, 'B65': 0.848},
            intention=1,
            resis_size=82,
            Ri=2.4,
            production_change_time=30,
            eff_day=28,
            full_vacc_size=40,
            defect_vac = 0.01
    ):
        self.p = p
        self.num_nodes = num_nodes
        prob = avg_node_degree / self.num_nodes
        self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=prob, seed=seeds['graph'])
        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.initial_outbreak_size = (
            initial_outbreak_size if initial_outbreak_size <= num_nodes else num_nodes
        )
        # self.virus_spread_chance = virus_spread_chance
        self.recovery_chance = recovery_chance
        self.death_chance = death_chance
        self.vac_effective = vac_effective
        self.gain_resistance_chance_after_infected = gain_resistance_chance_after_infected
        self.hosp_rate = hosp_rate
        self.area_ratio = area_ratio
        self.type_ratio = type_ratio
        self.intention = intention
        self.resis_size = resis_size
        self.eff_day = eff_day
        self.virus_spread_chance = dict([('s_a_above65_city', Ri / 2.4 * 0.05), ('s_a_below65_city', Ri / 2.4 * 0.01),
                                         ('s_a_above65_rural', Ri / 2.4 * 0.02), \
                                         ('s_a_below65_rural', Ri / 2.4 * 0.004), ('d_a_above65', Ri / 2.4 * 0.01),
                                         ('d_a_below65', Ri / 2.4 * 0.002)])
        self.full_vacc_size = full_vacc_size
        self.defect_vac = defect_vac
        self.datacollector = DataCollector(
            {
                "Infected_vac": number_infected_vac,
                "Infected_not_vac": number_infected_not_vac,
                "city_sus_above65": number_city_susceptible_above65,
                "city_sus_below65": number_city_susceptible_below65,
                'rural_sus_above65': number_rural_susceptible_above65,
                'rural_sus_below65': number_rural_susceptible_below65,
                "Resistant_vac": number_resistant_vac,
                "Resistant_nature": number_resistant_nature,
                "Death": number_death,
                'Vccinated_state': number_vac,
                'State_VACC_NO': number_vac_no,
                'State_VACC_YES': number_vac_yes,
                'accumulated_infected_number': number_accumulate_infected
            }
        )

        # Create agents 城市乡下 65上下划分 该有4类
        z_lists = [0] * self.num_nodes
        random.seed(seeds['dist_1'])
        city_agents_id = random.sample(range(0, self.num_nodes), int(self.area_ratio['city'] * self.num_nodes))
        for z in range(len(city_agents_id)):
            z_lists[city_agents_id[z]] = 1  # city编号为1
        rural_agents_id = [k for k, x in enumerate(z_lists) if x == 0]
        random.seed(seeds['dist_2'])
        rural_B65_agents_id = random.sample(rural_agents_id, int(self.type_ratio['B65'] * len(rural_agents_id)))
        for z in range(len(rural_B65_agents_id)):
            z_lists[rural_B65_agents_id[z]] = 2  # rural_b65编号为2 rural_a65为0
        random.seed(seeds['dist_3'])
        city_B65_agents_id = random.sample(city_agents_id, int(self.type_ratio['B65'] * len(city_agents_id)))
        for z in range(len(city_B65_agents_id)):
            z_lists[city_B65_agents_id[z]] = 3  # city_b65编号为3 city_a65为1
        distribution = Counter(z_lists)
        for i, node in enumerate(self.G.nodes()):
            if z_lists[i] == 0:
                a = VirusAgent(
                    i,
                    self,
                    State.SUS,
                    'rural',
                    'A65',
                    self.virus_spread_chance,
                    self.death_chance,
                    self.recovery_chance,
                    self.gain_resistance_chance_after_infected,
                    self.vac_effective,
                    self.hosp_rate,
                    self.num_nodes,
                    self.area_ratio,
                    self.type_ratio,
                    distribution,
                    intention=self.intention,
                    p=self.p,
                    production_change_time=production_change_time,
                    eff_day=self.eff_day,
                    defect_vac=self.defect_vac
                )
            elif z_lists[i] == 1:
                a = VirusAgent(
                    i,
                    self,
                    State.SUS,
                    'city',
                    'A65',
                    self.virus_spread_chance,
                    self.death_chance,
                    self.recovery_chance,
                    self.gain_resistance_chance_after_infected,
                    self.vac_effective,
                    self.hosp_rate,
                    self.num_nodes,
                    self.area_ratio,
                    self.type_ratio,
                    distribution,
                    intention=self.intention,
                    p=self.p,
                    production_change_time=production_change_time,
                    eff_day=self.eff_day,
                    defect_vac=self.defect_vac
                )
            elif z_lists[i] == 2:
                a = VirusAgent(
                    i,
                    self,
                    State.SUS,
                    'rural',
                    'B65',
                    self.virus_spread_chance,
                    self.death_chance,
                    self.recovery_chance,
                    self.gain_resistance_chance_after_infected,
                    self.vac_effective,
                    self.hosp_rate,
                    self.num_nodes,
                    self.area_ratio,
                    self.type_ratio,
                    distribution,
                    intention=self.intention,
                    p=self.p,
                    production_change_time=production_change_time,
                    eff_day=self.eff_day,
                    defect_vac=self.defect_vac
                )
            elif z_lists[i] == 3:
                a = VirusAgent(
                    i,
                    self,
                    State.SUS,
                    'city',
                    'B65',
                    self.virus_spread_chance,
                    self.death_chance,
                    self.recovery_chance,
                    self.gain_resistance_chance_after_infected,
                    self.vac_effective,
                    self.hosp_rate,
                    self.num_nodes,
                    self.area_ratio,
                    self.type_ratio,
                    distribution,
                    intention=self.intention,
                    p=self.p,
                    production_change_time=production_change_time,
                    eff_day=self.eff_day,
                    defect_vac=self.defect_vac
                )
            self.schedule.add(a)
            # Add the agent to the node
            self.grid.place_agent(a, node)

        # Infect some nodes
        random.seed(seeds['infect_id'])
        infected_nodes = self.random.sample(self.G.nodes(), self.initial_outbreak_size + self.resis_size+
                                            self.full_vacc_size)
        # inrs_lists = [1] * (self.initial_outbreak_size + self.resis_size)
        # res_id = random.sample(range(0, self.initial_outbreak_size + self.resis_size), self.resis_size)
        # for k in range(len(res_id)):
        #     inrs_lists[res_id[k]] = 0  # res编号为0
        rr = 0
        for a in self.grid.get_cell_list_contents(infected_nodes):
            if rr < self.initial_outbreak_size:
                a.state = State.NOT_VACCINATED_INFECTED
            elif self.initial_outbreak_size <= rr < self.initial_outbreak_size + self.resis_size:
                a.state = State.INFECTED_RECOVERED_RESISTANT
            else:
                a.state = State.VACC_yes
            rr += 1
            a.intention_to_vacc = 2

        self.running = True
        self.datacollector.collect(self)

    def resistant_susceptible_ratio(self):
        # try:
        #     return number_state(self, State.RESISTANT) / number_state(
        #         self, State.SUSCEPTIBLE
        #     )
        # except ZeroDivisionError:
        return math.inf

    def step(self):
        self.schedule.step()
        # collect data
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in tqdm(range(n)):
            self.step()


class VirusAgent(Agent):
    def __init__(
            self,
            unique_id,
            model,
            initial_state,
            area,
            sus_type,
            virus_spread_chance,
            death_chance,
            recovery_chance,
            gain_resistance_chance_after_infected,
            vac_effective,
            hosp_rate,
            agent_number,
            area_ratio,
            type_ratio,
            distribution,
            time_record_infected=0,
            time_record_vacc=0,
            time_record_check=0,
            production_vacc=0,
            vacc_state=0,
            intention=1,
            p=1,
            eff_day=28,
            production_change_time=30,
            defect_vac = 0.01
    ):
        super().__init__(unique_id, model)

        self.state = initial_state
        self.area = area
        self.sus_type = sus_type
        self.virus_spread_chance = virus_spread_chance
        self.death_chance = death_chance
        self.recovery_chance = recovery_chance
        self.vac_effective = vac_effective
        self.ini_vac_effective = vac_effective

        self.gain_resistance_chance_after_infected = gain_resistance_chance_after_infected
        self.hosp_rate = hosp_rate
        self.agent_number = agent_number
        self.area_ratio = area_ratio
        self.type_ratio = type_ratio
        self.distribution = distribution
        self.time_record_infected = time_record_infected
        self.time_record_vacc = time_record_vacc
        self.time_record_check = time_record_check
        self.production_vacc = production_vacc
        self.vacc_state = vacc_state
        self.intention_to_vacc = 0
        self.intention = intention
        self.p = p
        self.effday = eff_day
        self.p_c_time = production_change_time
        self.defect_vac = defect_vac
        # self.take_effect_time_gap = take_effect_time_gap

    def try_to_infect_neighbors(self):
        neighbors_nodes = self.model.grid.get_neighbors(self.pos, include_center=False)
        # virus_spread_chance={'s_a_above65': 0.15, 's_a_below65': 0.08, 'd_a_above':0.1, 'd_a_below':0.05},
        susceptible_neighbors = [
            agent
            for agent in self.model.grid.get_cell_list_contents(neighbors_nodes)
            if agent.state is State.SUS or agent.state is State.VACC_no
        ]
        for a in susceptible_neighbors:
            if self.area == 'city':
                if a.sus_type == 'A65' and a.area == 'city':
                    if self.random.random() < self.virus_spread_chance['s_a_above65_city']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'B65' and a.area == 'city':
                    if self.random.random() < self.virus_spread_chance['s_a_below65_city']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'A65' and a.area == 'rural':
                    if self.random.random() < self.virus_spread_chance['d_a_above65']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'B65' and a.area == 'rural':
                    if self.random.random() < self.virus_spread_chance['d_a_below65']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
            if self.area == 'rural':
                if a.sus_type == 'A65' and a.area == 'city':
                    if self.random.random() < self.virus_spread_chance['d_a_above65']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'B65' and a.area == 'city':
                    if self.random.random() < self.virus_spread_chance['d_a_below65']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'A65' and a.area == 'rural':
                    if self.random.random() < self.virus_spread_chance['s_a_above65_rural']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time
                if a.sus_type == 'B65' and a.area == 'rural':
                    if self.random.random() < self.virus_spread_chance['s_a_below65_rural']:
                        a.state = State.NOT_VACCINATED_INFECTED
                        a.time_record_infected = self.model.schedule.time

        vacc_yes_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
                              agent.state is State.VACC_yes]
        for a in vacc_yes_neighbors:
            if self.random.random() < self.vac_effective:
                a.state = State.VACCINATED_EFFECTIVE_RESISTANT
            else:
                a.state = State.VACCINATED_NOT_EFFECTIVE_INFECTED

        # vacc_no_neighbors = [agent for agent in self.model.grid.get_cell_list_contents(neighbors_nodes) if
        #                   agent.state is State.VACC_no]

        # for a in vacc_no_neighbors:
        #     if self.area == 'city':
        #         if a.sus_type == 'A65' and a.area == 'city':
        #             if self.random.random() < self.virus_spread_chance['s_a_above65']:
        #                 a.state = State.NOT_VACCINATED_INFECTED
        #         if a.sus_type == 'A65' and a.area == 'city':
        #             if self.random.random() < self.virus_spread_chance['s_a_above65']:
        #                 a.state = State.NOT_VACCINATED_INFECTED

    # 疫苗分配我们先考虑城市极端分配
    def get_vacc(self):
        if self.state == State.SUS:
            if number_vac(self.model) < self.production_vacc:
                if self.sus_type == 'A65' and self.area == 'city':
                    self.state = State.VACC_no
                    self.time_record_vacc = self.model.schedule.time
                    self.vacc_state = 1
                elif self.sus_type == 'B65' and self.area == 'city':
                    if number_city_susceptible_above65(self.model) <= number_intention_initial(self.model, 2):
                        self.state = State.VACC_no
                        self.time_record_vacc = self.model.schedule.time
                        self.vacc_state = 1
                elif self.sus_type == 'A65' and self.area == 'rural':
                    if number_city_susceptible_below65(self.model) <= number_intention_initial(self.model, 2,
                                                                                               ['city', 'B65']):
                        self.state = State.VACC_no
                        self.time_record_vacc = self.model.schedule.time
                        self.vacc_state = 1
                elif self.sus_type == 'B65' and self.area == 'rural':
                    if number_rural_susceptible_above65(self.model) <= number_intention_initial(self.model, 2,
                                                                                                ['rural', 'A65']):
                        self.state = State.VACC_no
                        self.time_record_vacc = self.model.schedule.time
                        self.vacc_state = 1

        elif self.state == State.VACC_no:
            if self.model.schedule.time - self.time_record_vacc >= self.effday:
                self.state = State.VACC_yes
        elif self.state == State.VACC_yes:
            past_time = self.model.schedule.time - self.time_record_vacc - self.effday
            self.vac_effective = self.ini_vac_effective*(1 - past_time*self.defect_vac)
        else:
            pass

    def try_gain_resistance(self):
        if self.state == State.VACCINATED_NOT_EFFECTIVE_INFECTED or self.state == State.NOT_VACCINATED_INFECTED:
            if self.model.schedule.time - self.time_record_infected >= 28:
                if self.random.random() < self.death_chance:
                    self.state = State.DEATH
                else:
                    self.state = State.INFECTED_RECOVERED_RESISTANT

    def try_check_situation(self):
        # if self.random.random()<0.7:
        # Checking...
        self.get_vacc()
        self.try_gain_resistance()
        # self.time_record_check = self.model.schedule.time

    def step(self):
        if self.model.schedule.time <= self.p_c_time:
            inc = int(self.p * self.agent_number)
        else:
            inc = int(self.p * self.agent_number)
        self.production_vacc += inc
        if self.state == State.VACCINATED_NOT_EFFECTIVE_INFECTED or self.state == State.NOT_VACCINATED_INFECTED:
            self.try_to_infect_neighbors()
        if self.intention_to_vacc == 0:
            if self.random.random() <= self.intention:
                self.intention_to_vacc = 1
            else:
                self.intention_to_vacc = 2

        if self.intention_to_vacc == 1:
            self.get_vacc()
        self.try_gain_resistance()
