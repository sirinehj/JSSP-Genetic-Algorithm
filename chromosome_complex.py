"""
Job Shop Scheduling - Algorithme Génétique OPTIMISÉ
Version simplifiée pour grands datasets
"""

import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import numpy as np
import os
import time
import sys
import gc
from collections import defaultdict

# ==================== CONFIGURATION ====================
DATA_DIR = "data"
RESULTS_DIR = "Results"

for directory in [DATA_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==================== CHARGEMENT DES DONNÉES ====================
def load_tasks(filename='tasks_large.json'):
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
    for task in tasks:
        task_by_id[task['id']] = task
    return task_by_id

# ==================== REPRÉSENTATION ====================
class OperationGene:
    """Représente une opération"""
    __slots__ = ('job_id', 'op_index', 'machine_id', 'duration')
    
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
    """Représente une solution"""
    def __init__(self, genes=None, num_jobs=0):
        self.genes = genes if genes else []
        self.fitness = 0
        self.makespan = float('inf')
        self.num_jobs = num_jobs
    
    def __repr__(self):
        return f"Chromosome(makespan={self.makespan:.0f})"

# ==================== DÉCODAGE CORRECT ====================
def decode_chromosome(chromosome, tasks, task_by_id):
    """
    Décodage CORRECT qui reproduit le comportement du décodeur original
    Ce décodeur parcourt le chromosome séquentiellement et planifie les opérations
    dans l'ordre où elles apparaissent, en respectant les contraintes de précédence
    """
    # Initialisation des structures
    machine_end_times = defaultdict(int)  # temps de fin sur chaque machine
    job_end_times = defaultdict(int)      # temps de fin du dernier opération de chaque job
    job_next_op = defaultdict(int)        # prochaine opération à planifier pour chaque job
    
    schedule = []
    scheduled_ops = set()
    total_ops = sum(len(task['operations']) for task in tasks)
    
    # Parcourir le chromosome plusieurs fois jusqu'à ce que toutes les opérations soient planifiées
    # Ceci reproduit la logique de la boucle while du décodeur original
    while len(scheduled_ops) < total_ops:
        for gene in chromosome.genes:
            if len(scheduled_ops) >= total_ops:
                break
                
            job_id = gene.job_id
            op_idx = gene.op_index
            op_key = (job_id, op_idx)
            
            # Skip si déjà planifié
            if op_key in scheduled_ops:
                continue
                
            # Vérifier si c'est la prochaine opération à planifier pour ce job
            if op_idx != job_next_op[job_id]:
                continue
                
            # Récupérer les données de l'opération
            task = task_by_id[job_id]
            op_data = task['operations'][op_idx]
            machine_id = op_data['machine_id']
            duration = op_data['duration']
            
            # Temps de début = max(fin précédente opération du job, fin dernière opération sur la machine)
            job_ready_time = job_end_times[job_id]
            machine_ready_time = machine_end_times[machine_id]
            start_time = max(job_ready_time, machine_ready_time)
            end_time = start_time + duration
            
            # Planifier l'opération
            schedule.append({
                'job_id': job_id,
                'op_idx': op_idx,
                'machine_id': machine_id,
                'start': start_time,
                'end': end_time,
                'duration': duration
            })
            
            # Mettre à jour les états
            scheduled_ops.add(op_key)
            job_next_op[job_id] += 1
            job_end_times[job_id] = end_time
            machine_end_times[machine_id] = end_time
    
    # Calculer le makespan (temps de fin maximum parmi tous les jobs)
    makespan = max(job_end_times.values()) if job_end_times else 0
    
    # Vérification
    if len(scheduled_ops) != total_ops:
        print(f"⚠️ ATTENTION: {len(scheduled_ops)}/{total_ops} opérations planifiées")
    
    return schedule, makespan

# ==================== INITIALISATION ====================
def create_initial_population(tasks, population_size=20):
    """Crée une population initiale"""
    print(f"Création de la population ({population_size} individus)...")
    
    # Créer toutes les opérations
    all_operations = []
    for task in tasks:
        task_id = task['id']
        for op_idx, op_data in enumerate(task['operations']):
            all_operations.append(OperationGene(
                job_id=task_id,
                op_index=op_idx,
                machine_id=op_data['machine_id'],
                duration=op_data['duration']
            ))
    
    total_ops = len(all_operations)
    print(f"  Total opérations: {total_ops:,}")
    
    # Créer la population
    population = []
    for i in range(population_size):
        genes = all_operations.copy()
        random.shuffle(genes)
        chromosome = Chromosome(genes, len(tasks))
        population.append(chromosome)
        
        if (i + 1) % 5 == 0:
            print(f"  Créé {i + 1}/{population_size} individus...")
    
    return population

# ==================== OPÉRATEURS GÉNÉTIQUES ====================
def tournament_selection(population, tournament_size=3):
    """Sélection par tournoi"""
    if len(population) < tournament_size:
        return random.choice(population)
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

def order_crossover(parent1, parent2):
    """Order Crossover (OX) pour les permutations"""
    size = len(parent1.genes)
    if size < 2:
        return deepcopy(parent1), deepcopy(parent2)
    
    # Choisir un segment (limité à 100 gènes max pour la performance)
    segment_size = min(50, size // 20)
    point1 = random.randint(0, size - segment_size)
    point2 = point1 + segment_size
    
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
    
    return Chromosome(child1_genes, parent1.num_jobs), Chromosome(child2_genes, parent2.num_jobs)

def swap_mutation(chromosome, mutation_rate=0.1):
    """Mutation par échange"""
    if random.random() > mutation_rate or len(chromosome.genes) < 2:
        return chromosome
    
    # Échanger deux positions aléatoires
    pos1, pos2 = random.sample(range(len(chromosome.genes)), 2)
    chromosome.genes[pos1], chromosome.genes[pos2] = chromosome.genes[pos2], chromosome.genes[pos1]
    return chromosome

# ==================== ALGORITHME GÉNÉTIQUE ====================
def genetic_algorithm_large(tasks, population_size=20, num_generations=30):
    """Algorithme génétique optimisé pour grands datasets"""
    print(f"\n{'='*60}")
    print(f"ALGORITHME GÉNÉTIQUE POUR GRAND DATASET")
    print(f"{'='*60}")
    
    # Statistiques
    total_ops = sum(len(task['operations']) for task in tasks)
    print(f"Jobs: {len(tasks):,}")
    print(f"Opérations totales: {total_ops:,}")
    print(f"Population: {population_size}")
    print(f"Générations: {num_generations}")
    
    # Préparer les données
    task_by_id = create_task_mapping(tasks)
    
    # Créer la population initiale
    population = create_initial_population(tasks, population_size)
    
    # Évaluation initiale
    print("\nÉvaluation initiale...")
    for i, individual in enumerate(population):
        _, makespan = decode_chromosome(individual, tasks, task_by_id)
        individual.makespan = makespan
        individual.fitness = 1.0 / makespan if makespan > 0 else 0
        
        if (i + 1) % 5 == 0:
            print(f"  Évalué {i + 1}/{len(population)} individus...")
    
    # Exécution principale
    print("\nDémarrage de l'évolution...")
    start_time = time.time()
    best_solution = None
    
    for generation in range(num_generations):
        # Trier par fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        current_best = population[0]
        
        # Sauvegarder la meilleure solution
        if best_solution is None or current_best.fitness > best_solution.fitness:
            best_solution = deepcopy(current_best)
        
        # Afficher la progression
        if generation % 5 == 0 or generation == num_generations - 1:
            elapsed = time.time() - start_time
            progress = (generation + 1) / num_generations * 100
            print(f"Gen {generation:3d}: Makespan = {current_best.makespan:8.0f}, "
                  f"Progress = {progress:5.1f}%, Time = {elapsed:.0f}s")
        
        # Nouvelle génération (élitisme)
        elite_count = max(2, int(population_size * 0.1))
        new_population = deepcopy(population[:elite_count])
        
        # Reproduction
        while len(new_population) < population_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover (80% de chance)
            if random.random() < 0.8:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = deepcopy(parent1), deepcopy(parent2)
            
            # Mutation
            child1 = swap_mutation(child1, mutation_rate=0.1)
            child2 = swap_mutation(child2, mutation_rate=0.1)
            
            # Ajouter à la nouvelle population
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        # Réévaluation des nouveaux individus seulement
        population = new_population[:population_size]
        for i in range(elite_count, len(population)):
            _, makespan = decode_chromosome(population[i], tasks, task_by_id)
            population[i].makespan = makespan
            population[i].fitness = 1.0 / makespan if makespan > 0 else 0
        
        # Nettoyage mémoire périodique
        if generation % 10 == 0:
            gc.collect()
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS FINAUX")
    print(f"{'='*60}")
    print(f"Temps total: {total_time:.1f} secondes")
    print(f"Meilleur makespan: {best_solution.makespan:,.0f}")
    print(f"Fitness: {best_solution.fitness:.6f}")
    print(f"{'='*60}")
    
    return best_solution

# ==================== VALIDATION ====================
def validate_solution(chromosome, tasks):
    """Validation simplifiée"""
    print("\n=== VALIDATION DE LA SOLUTION ===")
    
    task_by_id = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    # 1. Vérifier le nombre d'opérations
    expected_ops = sum(len(task['operations']) for task in tasks)
    actual_ops = len(schedule)
    
    print(f"1. Opérations attendues: {expected_ops:,}")
    print(f"   Opérations planifiées: {actual_ops:,}")
    
    if expected_ops != actual_ops:
        print(f"   ❌ ERREUR: {expected_ops - actual_ops} opérations manquantes!")
        return False
    
    # 2. Vérifier les précédences sur un échantillon
    print(f"2. Vérification des précédences...")
    
    # Prendre un échantillon de jobs (max 100)
    sample_size = min(100, len(tasks))
    sample_jobs = random.sample(tasks, sample_size)
    sample_job_ids = {job['id'] for job in sample_jobs}
    
    # Regrouper les opérations par job
    job_ops = defaultdict(list)
    for op in schedule:
        if op['job_id'] in sample_job_ids:
            job_ops[op['job_id']].append(op)
    
    # Vérifier les précédences
    precedence_ok = True
    for job_id, ops in job_ops.items():
        ops.sort(key=lambda x: x['op_idx'])
        for i in range(1, len(ops)):
            if ops[i]['start'] < ops[i-1]['end']:
                print(f"   ❌ ERREUR précédence Job {job_id}")
                precedence_ok = False
                break
    
    if precedence_ok:
        print(f"   ✓ Toutes les précédences sont respectées (sur {sample_size} jobs)")
    
    # 3. Statistiques de base
    print(f"\n3. Statistiques de la solution:")
    print(f"   • Makespan: {makespan:,.0f}")
    
    # Calculer l'utilisation des machines
    machine_work = defaultdict(int)
    for op in schedule:
        machine_work[op['machine_id']] += op['duration']
    
    total_work = sum(machine_work.values())
    avg_utilization = total_work / (makespan * len(machine_work)) * 100 if makespan > 0 else 0
    
    print(f"   • Travail total: {total_work:,.0f}")
    print(f"   • Machines utilisées: {len(machine_work)}")
    print(f"   • Utilisation moyenne: {avg_utilization:.1f}%")
    
    return precedence_ok

# ==================== VISUALISATION ====================
def plot_gantt_sample(chromosome, tasks, sample_size=50, filename='gantt_sample.png'):
    """Génère un diagramme de Gantt pour un échantillon"""
    try:
        # Prendre un échantillon
        if len(tasks) > sample_size:
            sample_tasks = random.sample(tasks, sample_size)
        else:
            sample_tasks = tasks
        
        # Créer un chromosome pour l'échantillon
        task_by_id = {t['id']: t for t in sample_tasks}
        sample_genes = []
        
        for gene in chromosome.genes:
            if gene.job_id in task_by_id:
                sample_genes.append(gene)
        
        sample_chromosome = Chromosome(sample_genes, len(sample_tasks))
        
        # Décoder l'échantillon
        schedule, makespan = decode_chromosome(sample_chromosome, sample_tasks, task_by_id)
        
        # Créer le diagramme
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Couleurs
        colors = plt.cm.tab20(np.linspace(0, 1, len(sample_tasks)))
        job_to_index = {task['id']: i for i, task in enumerate(sample_tasks)}
        
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
            if duration > makespan * 0.03:
                label = str(job_id).replace('job_', 'J')
                ax.text(start + duration/2, y_pos, label,
                        ha='center', va='center', fontsize=8)
        
        # Configuration du graphique
        ax.set_xlim(0, makespan * 1.01)
        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.set_xlabel('Temps', fontsize=12)
        ax.set_ylabel('Machines', fontsize=12)
        ax.set_title(f'Diagramme de Gantt (Échantillon de {len(sample_tasks)} jobs) - Makespan: {makespan:.0f}', 
                     fontsize=14, pad=20)
        
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels([f'M{m}' for m in machines], fontsize=9)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Sauvegarder
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=200, bbox_inches='tight')
        plt.show()
        print(f"✓ Diagramme d'échantillon sauvegardé: {RESULTS_DIR}/{filename}")
        
    except Exception as e:
        print(f"⚠️  Erreur lors de la création du diagramme: {e}")

# ==================== FONCTION PRINCIPALE ====================
def main():
    """Fonction principale"""
    print("="*60)
    print("JOB SHOP SCHEDULING - GRAND DATASET")
    print("="*60)
    
    # Charger les données
    filename = 'tasks_large.json'
    tasks = load_tasks(filename)
    
    if not tasks:
        print("❌ Aucune tâche chargée. Vérifiez le fichier tasks_large.json")
        return
    
    # Afficher des statistiques
    print(f"\n📊 Statistiques du dataset:")
    print(f"  • Nombre de jobs: {len(tasks):,}")
    
    total_ops = sum(len(task['operations']) for task in tasks)
    avg_ops = total_ops / len(tasks)
    print(f"  • Total opérations: {total_ops:,}")
    print(f"  • Moyenne opérations/job: {avg_ops:.1f}")
    
    # Paramètres (ajustés pour le grand dataset)
    print(f"\n⚙️  Paramètres d'exécution:")
    print("  1. Test rapide (population=10, générations=10)")
    print("  2. Moyen (population=20, générations=30)")
    print("  3. Complet (population=30, générations=50)")
    
    choice = input("Choisissez une option (1-3, défaut=2): ").strip()
    
    if choice == '1':
        pop_size, num_gens = 10, 10
    elif choice == '3':
        pop_size, num_gens = 30, 50
    else:
        pop_size, num_gens = 20, 30
    
    # Exécuter l'algorithme
    print(f"\n🚀 Démarrage de l'algorithme génétique...")
    best_solution = genetic_algorithm_large(
        tasks,
        population_size=pop_size,
        num_generations=num_gens
    )
    
    # Validation
    print(f"\n🔍 Validation de la solution...")
    is_valid = validate_solution(best_solution, tasks)
    
    if is_valid:
        print(f"\n✅ SOLUTION VALIDE")
        print(f"   Makespan obtenu: {best_solution.makespan:,.0f}")
        
        # Générer un diagramme d'échantillon
        print(f"\n🎨 Génération d'un diagramme d'échantillon...")
        sample_size = min(50, len(tasks))
        plot_gantt_sample(best_solution, tasks, sample_size=sample_size)
        
        # Conseils pour amélioration
        print(f"\n💡 Conseils pour améliorer les résultats:")
        print(f"   • Augmenter la population (ex: 50)")
        print(f"   • Augmenter les générations (ex: 100)")
        print(f"   • Ajouter une recherche locale")
        print(f"   • Utiliser un crossover plus sophistiqué")
    else:
        print(f"\n❌ SOLUTION INVALIDE")

if __name__ == "__main__":
    main()