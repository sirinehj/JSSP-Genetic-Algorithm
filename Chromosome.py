import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import numpy as np
import os
import time

# ==================== CONFIGURATION DES CHEMINS ====================
DATA_DIR = "data"
RESULTS_DIR = "Results"

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Dossier '{directory}' créé")

# ==================== CHARGEMENT DES DONNÉES ====================
def load_tasks(filename='tasks.json'):
    """Charge les tâches depuis le fichier JSON"""
    # Construire le chemin relatif
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Fichier chargé: {filepath} ({len(data)} jobs)")
        return data
    except FileNotFoundError:
        # Ne pas afficher d'erreur pour le fichier par défaut s'il n'existe pas
        if filename != 'tasks.json':
            print(f"⚠️  Fichier non trouvé: {filepath}")
        return []  # Retourner une liste vide
    except json.JSONDecodeError:
        print(f"❌ Erreur de lecture JSON: {filepath}")
        return []

def create_task_mapping(tasks):
    """Crée un mapping entre job_id et index pour un accès rapide"""
    task_by_id = {}
    job_ids = []
    
    for idx, task in enumerate(tasks):
        job_id = task['id']
        task_by_id[job_id] = task
        job_ids.append(job_id)
    
    return task_by_id, job_ids

# ==================== REPRÉSENTATION DU CHROMOSOME ====================
class Chromosome:
    """Représente une solution (ordonnancement) pour le JSSP"""
    
    def __init__(self, genes=None, num_jobs=0):
        if genes is None:
            self.genes = []
        else:
            self.genes = genes
        self.fitness = 0
        self.makespan = float('inf')
        self.num_jobs = num_jobs
    
    def __repr__(self):
        return f"Chromosome(makespan={self.makespan:.2f}, fitness={self.fitness:.4f})"
    
    def is_valid(self, tasks):
        """Vérifie si le chromosome est valide"""
        expected_counts = {task['id']: len(task['operations']) for task in tasks}
        actual_counts = {}
        
        for gene in self.genes:
            if gene is None:
                return False, "Contient des gènes None"
            actual_counts[gene] = actual_counts.get(gene, 0) + 1
        
        for job_id, expected in expected_counts.items():
            if actual_counts.get(job_id, 0) != expected:
                return False, f"Job {job_id}: {actual_counts.get(job_id, 0)} au lieu de {expected}"
        
        return True, "Chromosome valide"

# ==================== INITIALISATION ====================
def create_initial_population(tasks, population_size):
    """Crée la population initiale avec des permutations aléatoires"""
    population = []
    
    all_operations = []
    for task in tasks:
        task_id = task['id']
        num_operations = len(task['operations'])
        for _ in range(num_operations):
            all_operations.append(task_id)
    
    print(f"Nombre total d'opérations: {len(all_operations)}")
    
    for i in range(population_size):
        genes = all_operations.copy()
        random.shuffle(genes)
        chromosome = Chromosome(genes, len(tasks))
        
        is_valid, msg = chromosome.is_valid(tasks)
        if not is_valid:
            print(f"ERREUR: Chromosome initial invalide: {msg}")
            chromosome = repair_chromosome(chromosome, tasks)
        
        population.append(chromosome)
    
    return population

# ==================== FONCTIONS DE RÉPARATION ====================
def repair_chromosome(chromosome, tasks):
    """Répare un chromosome invalide"""
    actual_counts = {}
    for gene in chromosome.genes:
        if gene is not None:
            actual_counts[gene] = actual_counts.get(gene, 0) + 1
    
    required_counts = {task['id']: len(task['operations']) for task in tasks}
    missing_genes = []
    
    for job_id, required in required_counts.items():
        actual = actual_counts.get(job_id, 0)
        if actual < required:
            missing_genes.extend([job_id] * (required - actual))
    
    repaired_genes = []
    missing_idx = 0
    
    for gene in chromosome.genes:
        if gene is None and missing_idx < len(missing_genes):
            repaired_genes.append(missing_genes[missing_idx])
            missing_idx += 1
        elif gene is not None:
            repaired_genes.append(gene)
    
    while missing_idx < len(missing_genes):
        repaired_genes.append(missing_genes[missing_idx])
        missing_idx += 1
    
    return Chromosome(repaired_genes, chromosome.num_jobs)

# ==================== DÉCODAGE ET ÉVALUATION ====================
def decode_chromosome(chromosome, tasks, task_by_id=None):
    """Décode le chromosome en ordonnancement et calcule le makespan"""
    if task_by_id is None:
        task_by_id, _ = create_task_mapping(tasks)
    
    job_operation_counter = {}
    machine_end_times = {}
    job_end_times = {}
    
    for task in tasks:
        job_id = task['id']
        job_operation_counter[job_id] = 0
        job_end_times[job_id] = 0
    
    schedule = []
    
    for gene in chromosome.genes:
        if gene is None:
            print(f"ATTENTION: Gène None trouvé dans le chromosome!")
            gene = tasks[0]['id']
        
        job_id = gene
        operation_index = job_operation_counter[job_id]
        
        task = task_by_id[job_id]
        operation = task['operations'][operation_index]
        machine_id = operation['machine_id']
        duration = operation['duration']
        
        if machine_id not in machine_end_times:
            machine_end_times[machine_id] = 0
        
        start_time = max(machine_end_times[machine_id], job_end_times[job_id])
        end_time = start_time + duration
        
        machine_end_times[machine_id] = end_time
        job_end_times[job_id] = end_time
        
        schedule.append({
            'job_id': job_id,
            'operation_index': operation_index,
            'machine_id': machine_id,
            'start': start_time,
            'end': end_time,
            'duration': duration
        })
        
        job_operation_counter[job_id] += 1
    
    makespan = max(job_end_times.values())
    return schedule, makespan

def evaluate_fitness(chromosome, tasks, task_by_id=None):
    """Évalue le fitness d'un chromosome"""
    is_valid, msg = chromosome.is_valid(tasks)
    if not is_valid:
        chromosome = repair_chromosome(chromosome, tasks)
    
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    chromosome.makespan = makespan
    chromosome.fitness = 1.0 / makespan if makespan > 0 else 0
    return chromosome.fitness

# ==================== OPÉRATEURS GÉNÉTIQUES ====================
def tournament_selection(population, tournament_size=3):
    """Sélection par tournoi"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

def order_crossover(parent1, parent2):
    """Order Crossover (OX) - robuste pour JSSP"""
    size = len(parent1.genes)
    point1 = random.randint(0, size - 2)
    point2 = random.randint(point1 + 1, size)
    
    child1 = [None] * size
    child2 = [None] * size
    
    for i in range(point1, point2):
        child1[i] = parent1.genes[i]
        child2[i] = parent2.genes[i]
    
    def fill_child(child, parent, segment_set):
        pos = point2
        for i in range(size):
            idx = (point2 + i) % size
            gene = parent.genes[idx]
            if gene not in segment_set:
                child[pos % size] = gene
                pos += 1
    
    fill_child(child1, parent2, set(parent1.genes[point1:point2]))
    fill_child(child2, parent1, set(parent2.genes[point1:point2]))
    
    # Correction des None
    for child in [child1, child2]:
        if None in child:
            all_genes = list(set(parent1.genes + parent2.genes))
            missing_genes = [g for g in all_genes if g not in child]
            for i in range(size):
                if child[i] is None and missing_genes:
                    child[i] = missing_genes.pop(0)
    
    return Chromosome(child1, parent1.num_jobs), Chromosome(child2, parent2.num_jobs)

def swap_mutation(chromosome):
    """Mutation par échange de deux gènes"""
    size = len(chromosome.genes)
    if size < 2:
        return chromosome
    
    pos1 = random.randint(0, size - 1)
    pos2 = random.randint(0, size - 1)
    
    while pos2 == pos1:
        pos2 = random.randint(0, size - 1)
    
    chromosome.genes[pos1], chromosome.genes[pos2] = \
        chromosome.genes[pos2], chromosome.genes[pos1]
    
    return chromosome

# ==================== ALGORITHME GÉNÉTIQUE PRINCIPAL ====================
def genetic_algorithm(tasks, population_size=50, crossover_rate=0.8, 
                     mutation_rate=0.1, num_generations=100, elitism_count=2):
    """Algorithme génétique pour résoudre le JSSP"""
    
    print(f"\n{'='*60}")
    print(f"ALGORITHME GÉNÉTIQUE - JOB SHOP SCHEDULING")
    print(f"{'='*60}")
    print(f"Nombre de jobs: {len(tasks)}")
    print(f"Taille population: {population_size}")
    print(f"Générations: {num_generations}")
    
    total_operations = sum(len(task['operations']) for task in tasks)
    print(f"Total opérations: {total_operations}")
    
    task_by_id, job_ids = create_task_mapping(tasks)
    population = create_initial_population(tasks, population_size)
    
    for individual in population:
        evaluate_fitness(individual, tasks, task_by_id)
    
    best_fitness_history = []
    avg_fitness_history = []
    best_solution = None
    
    start_time = time.time()
    
    for generation in range(num_generations):
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = deepcopy(population[0])
        
        best_fitness = population[0].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        if generation % 10 == 0 or generation == num_generations - 1:
            elapsed = time.time() - start_time
            print(f"Gen {generation:4d}: Makespan = {population[0].makespan:8.2f}, "
                  f"Fitness = {best_fitness:.6f}, Time = {elapsed:.1f}s")
        
        new_population = deepcopy(population[:elitism_count])
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)
            
            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)
            
            is_valid1, _ = child1.is_valid(tasks)
            if not is_valid1:
                child1 = repair_chromosome(child1, tasks)
            
            is_valid2, _ = child2.is_valid(tasks)
            if not is_valid2:
                child2 = repair_chromosome(child2, tasks)
            
            if len(new_population) < population_size:
                new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population[:population_size]
        
        for individual in population:
            evaluate_fitness(individual, tasks, task_by_id)
    
    total_time = time.time() - start_time
    population.sort(key=lambda x: x.fitness, reverse=True)
    if population[0].fitness > best_solution.fitness:
        best_solution = deepcopy(population[0])
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Meilleur makespan: {best_solution.makespan:.2f}")
    print(f"Fitness: {best_solution.fitness:.6f}")
    print(f"Temps d'exécution: {total_time:.1f} secondes")
    print(f"{'='*60}")
    
    return best_solution, best_fitness_history, avg_fitness_history, job_ids

# ==================== VISUALISATION ET SAUVEGARDE ====================
def plot_gantt_chart(chromosome, tasks, job_ids, filename='gantt_chart.png'):
    """Génère un diagramme de Gantt pour la solution"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    task_by_id, _ = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    fig, ax = plt.subplots(figsize=(20, 12))
    
    job_to_index = {job_id: idx for idx, job_id in enumerate(job_ids)}
    
    num_jobs = len(job_ids)
    if num_jobs <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_jobs))
    
    machines = sorted(set(op['machine_id'] for op in schedule))
    machine_to_y = {machine: i for i, machine in enumerate(machines)}
    
    print(f"\nCréation du diagramme de Gantt...")
    print(f"Machines: {len(machines)}")
    print(f"Makespan: {makespan:.2f}")
    
    for operation in schedule:
        job_id = operation['job_id']
        job_index = job_to_index[job_id]
        machine_id = operation['machine_id']
        start = operation['start']
        duration = operation['duration']
        
        y_pos = machine_to_y[machine_id]
        
        rect = mpatches.Rectangle(
            (start, y_pos - 0.4), 
            duration, 
            0.8,
            facecolor=colors[job_index % len(colors)],
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
        
        if duration > makespan * 0.02:
            label = str(job_id).replace('job_', 'J')
            ax.text(start + duration/2, y_pos, label,
                    ha='center', va='center', fontsize=7, fontweight='bold')
    
    ax.set_xlim(0, makespan * 1.01)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    ax.set_xlabel('Temps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Machines', fontsize=14, fontweight='bold')
    ax.set_title(f'Diagramme de Gantt - Makespan: {makespan:.2f}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([str(m) for m in machines], fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.grid(axis='y', alpha=0.1, linestyle='-')
    
    legend_elements = []
    max_legend_jobs = min(15, len(job_ids))
    for i in range(max_legend_jobs):
        job_id = job_ids[i]
        label = str(job_id).replace('job_', 'Job ')
        legend_elements.append(
            mpatches.Patch(
                facecolor=colors[i % len(colors)], 
                edgecolor='black', 
                label=label,
                alpha=0.8
            )
        )
    
    if len(job_ids) > max_legend_jobs:
        legend_elements.append(
            mpatches.Patch(
                facecolor='gray', 
                edgecolor='black', 
                label=f'... et {len(job_ids) - max_legend_jobs} autres jobs',
                alpha=0.5
            )
        )
    
    ax.legend(handles=legend_elements, loc='upper right', 
             bbox_to_anchor=(1.15, 1), fontsize=9, title="Jobs")
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Diagramme sauvegardé: {filepath}")

def plot_convergence(best_fitness_history, avg_fitness_history, 
                     filename='convergence.png'):
    """Trace la convergence de l'algorithme"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    plt.figure(figsize=(12, 7))
    generations = range(len(best_fitness_history))
    
    plt.plot(generations, best_fitness_history, 'b-', 
             label='Meilleur Fitness', linewidth=2.5, alpha=0.8)
    plt.plot(generations, avg_fitness_history, 'r--', 
             label='Fitness Moyen', linewidth=2, alpha=0.7)
    
    plt.xlabel('Génération', fontsize=13, fontweight='bold')
    plt.ylabel('Fitness (1/Makespan)', fontsize=13, fontweight='bold')
    plt.title('Convergence de l\'Algorithme Génétique - JSSP', 
              fontsize=15, fontweight='bold', pad=15)
    
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ Convergence sauvegardée: {filepath}")

def save_solution_stats(chromosome, tasks, filename='solution_stats.txt'):
    """Sauvegarde les statistiques de la solution"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    task_by_id, _ = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    machine_utilization = {}
    job_completion_times = {}
    
    for op in schedule:
        machine_id = op['machine_id']
        job_id = op['job_id']
        duration = op['duration']
        
        if machine_id not in machine_utilization:
            machine_utilization[machine_id] = 0
        machine_utilization[machine_id] += duration
        
        job_completion_times[job_id] = max(job_completion_times.get(job_id, 0), op['end'])
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STATISTIQUES DE LA SOLUTION JSSP\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"MAKESPAN: {makespan:.2f}\n")
        f.write(f"FITNESS: {chromosome.fitness:.6f}\n")
        f.write(f"NOMBRE DE JOBS: {chromosome.num_jobs}\n")
        f.write(f"NOMBRE D'OPÉRATIONS: {len(schedule)}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("UTILISATION DES MACHINES\n")
        f.write("-"*60 + "\n")
        for machine_id in sorted(machine_utilization.keys()):
            utilization = (machine_utilization[machine_id] / makespan) * 100
            f.write(f"{machine_id}: {utilization:.1f}% ({machine_utilization[machine_id]:.1f}/{makespan:.1f})\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("TEMPS DE FIN DES JOBS (20 premiers)\n")
        f.write("-"*60 + "\n")
        sorted_jobs = sorted(job_completion_times.items(), key=lambda x: x[1])
        for job_id, completion_time in sorted_jobs[:20]:
            f.write(f"{job_id}: {completion_time:.2f}\n")
        if len(job_completion_times) > 20:
            f.write(f"\n... et {len(job_completion_times) - 20} autres jobs\n")
    
    print(f"✓ Statistiques sauvegardées: {filepath}")