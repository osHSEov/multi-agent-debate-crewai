from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from langchain_ollama import OllamaLLM
import os
import yaml

from crewai_tools import SerperDevTool



@CrewBase
class DebateSystem():
    

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def scientist(self) -> Agent:
        return Agent(
            config=self.agents_config['scientist'],
            llm=os.getenv("MODEL", "llama3.3:70b"),
            tools=[SerperDevTool()],
            max_iter=5,
            verbose=True
        )
    @agent
    def skeptic(self) -> Agent:
        return Agent(
            config=self.agents_config['skeptic'],
            llm=os.getenv("MODEL", "llama3.3:70b"),
            tools=[SerperDevTool()],
            max_iter=5,
            verbose=True
        )

    @agent
    def pragmatist(self) -> Agent:
        return Agent(
            config=self.agents_config['pragmatist'],
            llm=os.getenv("MODEL", "llama3.3:70b"),
            tools=[SerperDevTool()],
            max_iter=5,
            verbose=True
        )

    @agent
    def ethicist(self) -> Agent:
        return Agent(
            config=self.agents_config['ethicist'],
            llm=os.getenv("MODEL", "llama3.3:70b"),
            tools=[SerperDevTool()],
            max_iter=5,
            verbose=True
        )

    @agent
    def moderator(self) -> Agent:
        return Agent(
            config=self.agents_config['moderator'],
            llm=os.getenv("MODEL", "llama3.3:70b"),
            tools=[SerperDevTool()],
            max_iter=5,
            verbose=True
        )

    @task
    def scientist_task(self) -> Task:
        return Task(
            config=self.tasks_config['scientist_task'],
            agent=self.scientist()
        )

    @task
    def skeptic_task(self) -> Task:
        return Task(
            config=self.tasks_config['skeptic_task'],
            agent=self.skeptic(),
            context=[self.scientist_task()] 
        )

    @task
    def pragmatist_task(self) -> Task:
        return Task(
            config=self.tasks_config['pragmatist_task'],
            agent=self.pragmatist()
        )

    @task
    def ethicist_task(self) -> Task:
        return Task(
            config=self.tasks_config['ethicist_task'],
            agent=self.ethicist()
        )

    @task
    def moderator_task(self) -> Task:
        return Task(
            config=self.tasks_config['moderator_task'],
            agent=self.moderator(),
            output_file='final_report.md'
        )

    @crew
    def crew(self) -> Crew:
       
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
