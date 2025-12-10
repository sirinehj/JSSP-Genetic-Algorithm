"""
Job Shop Scheduling - Algorithme Génétique CORRIGÉ
Version avec représentation correcte des opérations
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import numpy as np
import os
import time

# ==================== CONFIGURATION ====================
DATA_DIR = "data"
RESULTS_DIR = "Results"

for directory in [DATA_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==================== CHARGEMENT DES DONNÉES ====================
def load_tasks(filename='tasks_small.json'):
    """Charge les tâches depuis le fichier JSON"""
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Fichier chargé: {filepath} ({len(data)} jobs)")
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Erreur: {e}")
        return []

def create_task_mapping(tasks):
    """Crée un mapping entre job_id et task"""
    task_by_id = {}
    job_ids = []
    
    for task in tasks:
        job_id = task['id']
        task_by_id[job_id] = task
        job_ids.append(job_id)
    
    return task_by_id, job_ids

# ==================== REPRÉSENTATION CORRECTE ====================
class OperationGene:
    """Représente une opération spécifique avec toutes ses propriétés"""
    def __init__(self, job_id, op_index, machine_id, duration):
        self.job_id = job_id
        self.op_index = op_index
        self.machine_id = machine_id
        self.duration = duration
    
    def __repr__(self):
        return f"({self.job_id}.{self.op_index}@{self.machine_id}:{self.duration})"
    
    def __eq__(self, other):
        if not isinstance(other, OperationGene):
            return False
        return (self.job_id == other.job_id and 
                self.op_index == other.op_index)
    
    def __hash__(self):
        return hash((self.job_id, self.op_index))

class Chromosome:
    """Représente une solution correcte"""
    def __init__(self, genes=None, num_jobs=0):
        if genes is None:
            self.genes = []  # Liste de OperationGene
        else:
            self.genes = genes
        self.fitness = 0
        self.makespan = float('inf')
        self.num_jobs = num_jobs
    
    def __repr__(self):
        return f"Chromosome(makespan={self.makespan:.2f}, fitness={self.fitness:.4f})"
    
    def is_valid(self, tasks):
        """Vérifie si le chromosome contient toutes les opérations"""
        expected_ops = set()
        for task in tasks:
            for op_idx in range(len(task['operations'])):
                expected_ops.add((task['id'], op_idx))
        
        actual_ops = set()
        for gene in self.genes:
            if isinstance(gene, OperationGene):
                actual_ops.add((gene.job_id, gene.op_index))
            else:
                return False, f"Gène invalide: {gene}"
        
        if len(actual_ops) != len(expected_ops):
            return False, f"Manque {len(expected_ops) - len(actual_ops)} opérations"
        
        if actual_ops != expected_ops:
            missing = expected_ops - actual_ops
            return False, f"Opérations manquantes: {missing}"
        
        return True, "Chromosome valide"

# ==================== FONCTIONS DE RÉPARATION ====================
def repair_chromosome(chromosome, tasks):
    """Répare un chromosome invalide"""
    # Récupérer toutes les opérations attendues
    expected_ops = []
    for task in tasks:
        task_id = task['id']
        for op_idx, op_data in enumerate(task['operations']):
            op_gene = OperationGene(
                job_id=task_id,
                op_index=op_idx,
                machine_id=op_data['machine_id'],
                duration=op_data['duration']
            )
            expected_ops.append(op_gene)
    
    # Gènes existants
    existing_genes = []
    for gene in chromosome.genes:
        if isinstance(gene, OperationGene):
            existing_genes.append(gene)
    
    # Trouver les gènes manquants
    existing_set = set(existing_genes)
    missing_genes = [g for g in expected_ops if g not in existing_set]
    
    # Combiner et compléter
    repaired_genes = existing_genes + missing_genes
    
    # S'assurer qu'on a toutes les opérations
    if len(repaired_genes) != len(expected_ops):
        # Ajouter les dernières manquantes
        for gene in expected_ops:
            if gene not in repaired_genes:
                repaired_genes.append(gene)
    
    return Chromosome(repaired_genes, chromosome.num_jobs)

# ==================== INITIALISATION ====================
def create_initial_population(tasks, population_size):
    """Crée la population avec la représentation correcte"""
    population = []
    
    # Créer toutes les opérations
    all_operations = []
    for task in tasks:
        task_id = task['id']
        for op_idx, op_data in enumerate(task['operations']):
            op_gene = OperationGene(
                job_id=task_id,
                op_index=op_idx,
                machine_id=op_data['machine_id'],
                duration=op_data['duration']
            )
            all_operations.append(op_gene)
    
    print(f"✓ Création de {len(all_operations)} opérations uniques")
    
    for i in range(population_size):
        genes = all_operations.copy()
        random.shuffle(genes)
        chromosome = Chromosome(genes, len(tasks))
        
        is_valid, msg = chromosome.is_valid(tasks)
        if not is_valid:
            print(f"⚠️ Chromosome {i} invalide: {msg}")
            chromosome = repair_chromosome(chromosome, tasks)
        
        population.append(chromosome)
    
    return population

# ==================== DÉCODAGE ET ÉVALUATION ====================
def decode_chromosome(chromosome, tasks, task_by_id=None):
    """Décode correctement le chromosome"""
    if task_by_id is None:
        task_by_id, _ = create_task_mapping(tasks)
    
    machine_end_times = {}
    job_end_times = {}
    job_next_op = {}
    job_last_end = {}
    
    # Initialiser pour chaque job
    for task in tasks:
        job_id = task['id']
        job_end_times[job_id] = 0
        job_next_op[job_id] = 0
        job_last_end[job_id] = 0
    
    schedule = []
    scheduled_ops = set()  # Pour suivre les opérations déjà planifiées
    
    # Fonction pour vérifier si une opération peut être planifiée
    def can_schedule(job_id, op_index):
        return op_index == job_next_op[job_id]
    
    # Continuer jusqu'à ce que toutes les opérations soient planifiées
    total_ops = sum(len(task['operations']) for task in tasks)
    
    while len(schedule) < total_ops:
        progress = False
        
        # Parcourir le chromosome
        for gene in chromosome.genes:
            if not isinstance(gene, OperationGene):
                continue
            
            job_id = gene.job_id
            op_idx = gene.op_index
            op_key = (job_id, op_idx)
            
            # Vérifier si déjà planifiée
            if op_key in scheduled_ops:
                continue
            
            # Vérifier si on peut planifier cette opération
            if not can_schedule(job_id, op_idx):
                continue
            
            # Récupérer les données de l'opération
            task = task_by_id[job_id]
            op_data = task['operations'][op_idx]
            machine_id = op_data['machine_id']
            duration = op_data['duration']
            
            # Initialiser la machine si nécessaire
            if machine_id not in machine_end_times:
                machine_end_times[machine_id] = 0
            
            # Calculer le temps de début
            # Dépend de la fin de l'opération précédente du job et de la disponibilité de la machine
            job_ready_time = job_last_end[job_id]
            machine_ready_time = machine_end_times[machine_id]
            start_time = max(job_ready_time, machine_ready_time)
            end_time = start_time + duration
            
            # Planifier l'opération
            schedule.append({
                'job_id': job_id,
                'operation_index': op_idx,
                'machine_id': machine_id,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
            
            # Mettre à jour les états
            scheduled_ops.add(op_key)
            job_next_op[job_id] += 1
            job_last_end[job_id] = end_time
            machine_end_times[machine_id] = end_time
            
            progress = True
            break  # Revenir au début de la boucle
        
        # Si aucune opération n'a pu être planifiée, forcer la planification
        if not progress:
            for gene in chromosome.genes:
                if not isinstance(gene, OperationGene):
                    continue
                
                job_id = gene.job_id
                op_idx = gene.op_index
                op_key = (job_id, op_idx)
                
                if op_key in scheduled_ops:
                    continue
                
                # Forcer la planification de cette opération
                task = task_by_id[job_id]
                op_data = task['operations'][op_idx]
                machine_id = op_data['machine_id']
                duration = op_data['duration']
                
                if machine_id not in machine_end_times:
                    machine_end_times[machine_id] = 0
                
                # Trouver le temps de fin du prédécesseur
                predecessor_end = 0
                for op in schedule:
                    if op['job_id'] == job_id and op['operation_index'] == op_idx - 1:
                        predecessor_end = op['end']
                        break
                
                start_time = max(machine_end_times[machine_id], predecessor_end)
                end_time = start_time + duration
                
                schedule.append({
                    'job_id': job_id,
                    'operation_index': op_idx,
                    'machine_id': machine_id,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
                
                scheduled_ops.add(op_key)
                job_next_op[job_id] = max(job_next_op[job_id], op_idx + 1)
                job_last_end[job_id] = max(job_last_end.get(job_id, 0), end_time)
                machine_end_times[machine_id] = end_time
                
                break
    
    # Calculer le makespan
    makespan = max(job_last_end.values()) if job_last_end else 0
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
    """Order Crossover (OX)"""
    size = len(parent1.genes)
    if size < 2:
        return deepcopy(parent1), deepcopy(parent2)
    
    point1 = random.randint(0, size - 2)
    point2 = random.randint(point1 + 1, size)
    
    child1_genes = [None] * size
    child2_genes = [None] * size
    
    # Copier les segments
    for i in range(point1, point2):
        child1_genes[i] = parent1.genes[i]
        child2_genes[i] = parent2.genes[i]
    
    # Remplir le reste
    def fill_child(child_genes, parent, segment_set):
        pos = point2
        for i in range(size):
            idx = (point2 + i) % size
            gene = parent.genes[idx]
            if gene not in segment_set:
                child_genes[pos % size] = gene
                pos += 1
    
    fill_child(child1_genes, parent2, set(parent1.genes[point1:point2]))
    fill_child(child2_genes, parent1, set(parent2.genes[point1:point2]))
    
    # Réparer si nécessaire
    child1 = Chromosome(child1_genes, parent1.num_jobs)
    child2 = Chromosome(child2_genes, parent2.num_jobs)
    
    return child1, child2

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

# ==================== ALGORITHME GÉNÉTIQUE ====================
def genetic_algorithm(tasks, population_size=50, crossover_rate=0.8, 
                     mutation_rate=0.1, num_generations=100, elitism_count=2):
    """Algorithme génétique pour résoudre le JSSP"""
    
    print(f"\n{'='*60}")
    print(f"ALGORITHME GÉNÉTIQUE - JOB SHOP SCHEDULING")
    print(f"{'='*60}")
    
    total_operations = sum(len(task['operations']) for task in tasks)
    print(f"Nombre de jobs: {len(tasks)}")
    print(f"Total opérations: {total_operations}")
    print(f"Taille population: {population_size}")
    print(f"Générations: {num_generations}")
    
    task_by_id, job_ids = create_task_mapping(tasks)
    population = create_initial_population(tasks, population_size)
    
    # Évaluation initiale
    for individual in population:
        evaluate_fitness(individual, tasks, task_by_id)
    
    best_fitness_history = []
    avg_fitness_history = []
    best_solution = None
    
    start_time = time.time()
    
    for generation in range(num_generations):
        # Trier par fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Mettre à jour la meilleure solution
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = deepcopy(population[0])
        
        # Historique
        best_fitness = population[0].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Affichage
        if generation % 10 == 0 or generation == num_generations - 1:
            elapsed = time.time() - start_time
            print(f"Gen {generation:4d}: Makespan = {population[0].makespan:8.2f}, "
                  f"Fitness = {best_fitness:.6f}, Time = {elapsed:.1f}s")
        
        # Nouvelle population (élitisme)
        new_population = deepcopy(population[:elitism_count])
        
        # Reproduction
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < crossover_rate:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)
            
            # Mutation
            if random.random() < mutation_rate:
                child1 = swap_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2)
            
            # Réparation si nécessaire
            is_valid1, _ = child1.is_valid(tasks)
            if not is_valid1:
                child1 = repair_chromosome(child1, tasks)
            
            is_valid2, _ = child2.is_valid(tasks)
            if not is_valid2:
                child2 = repair_chromosome(child2, tasks)
            
            # Ajout à la nouvelle population
            if len(new_population) < population_size:
                new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        population = new_population[:population_size]
        
        # Réévaluation
        for individual in population:
            evaluate_fitness(individual, tasks, task_by_id)
    
    total_time = time.time() - start_time
    
    # Vérifier la meilleure solution finale
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

# ==================== VALIDATION ====================
def validate_solution(chromosome, tasks):
    """Valide rigoureusement une solution"""
    print("\n=== VALIDATION DE LA SOLUTION ===")
    
    # 1. Vérifier la représentation
    is_valid, msg = chromosome.is_valid(tasks)
    print(f"1. Validité chromosome: {is_valid} ({msg})")
    
    if not is_valid:
        return False
    
    # 2. Décoder et vérifier les contraintes
    task_by_id, _ = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    print(f"2. Nombre d'opérations planifiées: {len(schedule)}")
    print(f"3. Makespan calculé: {makespan}")
    
    # 3. Vérifier toutes les opérations sont planifiées
    expected_count = sum(len(task['operations']) for task in tasks)
    if len(schedule) != expected_count:
        print(f"❌ ERREUR: {len(schedule)}/{expected_count} opérations planifiées")
        return False
    
    # 4. Vérifier les précédences
    job_ops = {}
    for op in schedule:
        job_id = op['job_id']
        if job_id not in job_ops:
            job_ops[job_id] = []
        job_ops[job_id].append(op)
    
    precedence_violation = False
    for job_id, ops in job_ops.items():
        ops.sort(key=lambda x: x['operation_index'])
        for i in range(1, len(ops)):
            if ops[i]['start'] < ops[i-1]['end']:
                print(f"❌ ERREUR précédence Job {job_id}: op{i-1} fin {ops[i-1]['end']}, op{i} début {ops[i]['start']}")
                precedence_violation = True
    
    if precedence_violation:
        return False
    
    print("✓ Solution VALIDE")
    return True

# ==================== VISUALISATION ====================
def plot_gantt_chart(chromosome, tasks, job_ids, filename='gantt_chart_genetic.png'):
    """Génère un diagramme de Gantt pour la solution"""
    try:
        task_by_id, _ = create_task_mapping(tasks)
        schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
        
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        fig, ax = plt.subplots(figsize=(20, 12))
        
        # Couleurs
        num_jobs = len(job_ids)
        if num_jobs <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
        else:
            colors = plt.cm.rainbow(np.linspace(0, 1, num_jobs))
        
        job_to_index = {job_id: idx for idx, job_id in enumerate(job_ids)}
        
        # Machines
        machines = sorted(set(op['machine_id'] for op in schedule))
        machine_to_y = {machine: i for i, machine in enumerate(machines)}
        
        # Dessiner les opérations
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
            
            # Texte si assez grand
            if duration > makespan * 0.02:
                label = str(job_id).replace('job_', 'J')
                ax.text(start + duration/2, y_pos, label,
                        ha='center', va='center', fontsize=7, fontweight='bold')
        
        ax.set_xlim(0, makespan * 1.01)
        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.set_xlabel('Temps', fontsize=14, fontweight='bold')
        ax.set_ylabel('Machines', fontsize=14, fontweight='bold')
        ax.set_title(f'Diagramme de Gantt - Makespan: {makespan:.2f} (Génétique)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels([str(m) for m in machines], fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Diagramme sauvegardé: {filepath}")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du diagramme: {e}")

# ==================== EXÉCUTION PRINCIPALE ====================
def main():
    """Fonction principale"""
    print("="*60)
    print("JOB SHOP SCHEDULING - ALGORITHME GÉNÉTIQUE")
    print("="*60)
    
    # Charger les données
    tasks = load_tasks('tasks_small.json')
    if not tasks:
        print("❌ Aucune tâche chargée. Vérifiez le fichier tasks.json")
        return
    
    # Exécuter l'algorithme génétique
    best_solution, best_history, avg_history, job_ids = genetic_algorithm(
        tasks,
        population_size=50,      # Taille raisonnable
        num_generations=100,     # Suffisant pour convergence
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_count=2
    )
    
    # Valider la solution
    is_valid = validate_solution(best_solution, tasks)
    
    if is_valid:
        print(f"\n✅ SOLUTION VALIDE: makespan = {best_solution.makespan:.2f}")
        print(f"   (À comparer avec Branch & Bound: 2843)")
        
        # Statistiques
        task_by_id, _ = create_task_mapping(tasks)
        schedule, _ = decode_chromosome(best_solution, tasks, task_by_id)
        
        machine_work = {}
        for op in schedule:
            machine = op['machine_id']
            machine_work[machine] = machine_work.get(machine, 0) + op['duration']
        
        total_work = sum(machine_work.values())
        print(f"\n📊 Statistiques:")
        print(f"   • Utilisation machines: {total_work / (best_solution.makespan * len(machine_work)) * 100:.1f}%")
        print(f"   • Durée totale travail: {total_work:.0f}")
        
        # Vérifier la cohérence
        total_work_branch = 8214  # De votre output Branch & Bound
        if abs(total_work - total_work_branch) > 10:
            print(f"⚠️  ATTENTION: Travail total différent")
            print(f"   • Génétique: {total_work:.0f}")
            print(f"   • Branch & Bound: {total_work_branch}")
        
        # Générer le diagramme de Gantt
        plot_gantt_chart(best_solution, tasks, job_ids)
        
    else:
        print(f"\n❌ SOLUTION INVALIDE")
        print("   Vérifiez le décodage du chromosome")

if __name__ == "__main__":
    main()