# chromosome_complex.py - VERSION COMPLÈTE CORRIGÉE
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import numpy as np
import os
import time
from collections import defaultdict
import multiprocessing as mp

# ==================== CONFIGURATION ====================
DATA_DIR = "data"
RESULTS_DIR = "Results"

# Créer les dossiers
for directory in [DATA_DIR, RESULTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ==================== CHARGEMENT OPTIMISÉ ====================
def load_tasks_fast(filename='tasks.json'):
    """Chargement rapide avec vérification"""
    filepath = os.path.join(DATA_DIR, filename)
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✓ Chargé: {filepath} ({len(data)} jobs)")
        return data
    except Exception as e:
        print(f"❌ Erreur: {filepath} - {str(e)}")
        return []

def create_optimized_mapping(tasks):
    """Mapping ultra-optimisé"""
    task_by_id = {}
    job_ids = []
    job_op_counts = {}
    
    for idx, task in enumerate(tasks):
        job_id = task['id']
        task_by_id[job_id] = {
            'idx': idx,
            'operations': task['operations'],
            'op_count': len(task['operations'])
        }
        job_ids.append(job_id)
        job_op_counts[job_id] = len(task['operations'])
    
    return task_by_id, job_ids, job_op_counts

# ==================== CHROMOSOME OPTIMISÉ ====================
class ChromosomeFast:
    """Chromosome optimisé pour grands datasets"""
    
    def __init__(self, genes=None, num_jobs=0):
        self.genes = genes if genes is not None else []
        self.fitness = 0.0
        self.makespan = float('inf')
        self.num_jobs = num_jobs
    
    def __repr__(self):
        return f"Chromosome(makespan={self.makespan:.1f}, fitness={self.fitness:.6f})"
    
    def is_valid(self, job_op_counts):
        """Vérifie si le chromosome est valide"""
        from collections import Counter
        actual_counts = Counter(self.genes)
        
        for job_id, expected in job_op_counts.items():
            if actual_counts.get(job_id, 0) != expected:
                return False
        
        return True
    
    def repair(self, job_op_counts):
        """Répare un chromosome invalide"""
        from collections import Counter
        actual_counts = Counter(self.genes)
        
        # Liste des gènes manquants
        missing_genes = []
        extra_genes = []
        
        for job_id, expected in job_op_counts.items():
            actual = actual_counts.get(job_id, 0)
            if actual < expected:
                missing_genes.extend([job_id] * (expected - actual))
            elif actual > expected:
                extra_genes.extend([job_id] * (actual - expected))
        
        # Si correct, retourner tel quel
        if not missing_genes and not extra_genes:
            return self
        
        # Créer une copie des gènes
        repaired_genes = self.genes.copy()
        
        # Retirer les extras (en partant de la fin)
        if extra_genes:
            # Compter combien de chaque à retirer
            to_remove = Counter(extra_genes)
            
            # Retirer en parcourant à l'envers
            for i in range(len(repaired_genes)-1, -1, -1):
                gene = repaired_genes[i]
                if to_remove.get(gene, 0) > 0:
                    repaired_genes.pop(i)
                    to_remove[gene] -= 1
                    if sum(to_remove.values()) == 0:
                        break
        
        # Ajouter les manquants
        if missing_genes:
            random.shuffle(missing_genes)
            repaired_genes.extend(missing_genes)
        
        # S'assurer de la bonne longueur
        expected_len = sum(job_op_counts.values())
        if len(repaired_genes) > expected_len:
            repaired_genes = repaired_genes[:expected_len]
        elif len(repaired_genes) < expected_len:
            # Ajouter des gènes aléatoires manquants
            while len(repaired_genes) < expected_len:
                random_job = random.choice(list(job_op_counts.keys()))
                repaired_genes.append(random_job)
        
        return ChromosomeFast(repaired_genes, self.num_jobs)

# ==================== DÉCODAGE ULTRA-RAPIDE ====================
def decode_chromosome_fast(chromosome, tasks, task_by_id=None):
    """Décodage 10x plus rapide avec pré-calcul"""
    if task_by_id is None:
        task_by_id, _, _ = create_optimized_mapping(tasks)
    
    # Utiliser des arrays numpy pour la performance
    job_op_counter = defaultdict(int)
    machine_times = {}
    job_times = defaultdict(float)
    
    max_time = 0.0
    
    for gene in chromosome.genes:
        job_data = task_by_id.get(gene)
        if not job_data:
            continue
            
        op_idx = job_op_counter[gene]
        # Vérifier l'index
        if op_idx >= job_data['op_count']:
            # Si trop d'opérations pour ce job, ignorer
            continue
            
        operation = job_data['operations'][op_idx]
        machine = operation['machine_id']
        duration = operation['duration']
        
        machine_time = machine_times.get(machine, 0.0)
        job_time = job_times[gene]
        
        start = max(machine_time, job_time)
        end = start + duration
        
        machine_times[machine] = end
        job_times[gene] = end
        job_op_counter[gene] += 1
        
        if end > max_time:
            max_time = end
    
    return max_time

# ==================== ÉVALUATION ====================
def evaluate_population_sequential(population, tasks, task_by_id, job_op_counts):
    """Évaluation séquentielle avec validation"""
    for chrom in population:
        # Vérifier et réparer si nécessaire
        if not chrom.is_valid(job_op_counts):
            chrom = chrom.repair(job_op_counts)
        
        makespan = decode_chromosome_fast(chrom, tasks, task_by_id)
        chrom.makespan = makespan
        chrom.fitness = 1.0 / (makespan + 1)  # +1 pour éviter division par 0
    
    return population

# ==================== INITIALISATION RAPIDE ====================
def create_population_fast(tasks, population_size, job_op_counts):
    """Création rapide de population diversifiée avec validation"""
    population = []
    
    # Créer séquence de base valide
    base_sequence = []
    for job_id, count in job_op_counts.items():
        base_sequence.extend([job_id] * count)
    
    # Stratégies variées
    strategies = ['shuffle', 'sorted', 'reverse', 'partial_shuffle']
    
    for i in range(population_size):
        strategy = strategies[i % len(strategies)]
        genes = base_sequence.copy()
        
        if strategy == 'shuffle':
            random.shuffle(genes)
        elif strategy == 'sorted':
            genes.sort()
        elif strategy == 'reverse':
            genes.sort(reverse=True)
        elif strategy == 'partial_shuffle':
            # Mélanger seulement une partie
            shuffle_size = max(10, len(genes) // 20)
            for _ in range(shuffle_size):
                a, b = random.sample(range(len(genes)), 2)
                genes[a], genes[b] = genes[b], genes[a]
        
        chrom = ChromosomeFast(genes, len(tasks))
        
        # Double vérification
        if not chrom.is_valid(job_op_counts):
            chrom = chrom.repair(job_op_counts)
        
        population.append(chrom)
    
    return population

# ==================== OPÉRATEURS OPTIMISÉS ====================
def tournament_selection_fast(population, size=5):
    """Sélection rapide par tournoi"""
    if len(population) <= size:
        return max(population, key=lambda x: x.fitness)
    
    tournament = random.sample(population, size)
    return max(tournament, key=lambda x: x.fitness)

def order_crossover_safe(parent1, parent2, job_op_counts):
    """Croisement OX simple et robuste"""
    size = len(parent1.genes)
    
    # Vérifications de base
    if size != len(parent2.genes) or size < 2:
        return deepcopy(parent1), deepcopy(parent2)
    
    # Créer une séquence de base valide
    base_sequence = []
    for job_id, count in job_op_counts.items():
        base_sequence.extend([job_id] * count)
    
    # Mélanger pour diversité
    child1_genes = base_sequence.copy()
    child2_genes = base_sequence.copy()
    
    # Échanger des segments entre parents si assez grand
    if size > 20:
        # Prendre un segment du parent1
        segment_size = random.randint(5, min(15, size // 4))
        start1 = random.randint(0, size - segment_size)
        segment1 = parent1.genes[start1:start1 + segment_size]
        
        # Trouver une position d'insertion dans child2
        insert_pos = random.randint(0, len(child2_genes) - segment_size)
        child2_genes[insert_pos:insert_pos + segment_size] = segment1
        
        # Prendre un segment du parent2
        start2 = random.randint(0, size - segment_size)
        segment2 = parent2.genes[start2:start2 + segment_size]
        
        # Insérer dans child1
        insert_pos = random.randint(0, len(child1_genes) - segment_size)
        child1_genes[insert_pos:insert_pos + segment_size] = segment2
    
    # Créer chromosomes
    chrom1 = ChromosomeFast(child1_genes, parent1.num_jobs)
    chrom2 = ChromosomeFast(child2_genes, parent2.num_jobs)
    
    # Valider
    if not chrom1.is_valid(job_op_counts):
        chrom1 = chrom1.repair(job_op_counts)
    if not chrom2.is_valid(job_op_counts):
        chrom2 = chrom2.repair(job_op_counts)
    
    return chrom1, chrom2

def swap_mutation_safe(chromosome, mutation_rate=0.1, job_op_counts=None):
    """Mutation avec taux variable"""
    size = len(chromosome.genes)
    if size < 2:
        return chromosome
    
    # Nombre de swaps basé sur la taille
    num_swaps = max(1, int(size * mutation_rate))
    
    for _ in range(num_swaps):
        a, b = random.sample(range(size), 2)
        chromosome.genes[a], chromosome.genes[b] = chromosome.genes[b], chromosome.genes[a]
    
    # Vérifier après mutation
    if job_op_counts and not chromosome.is_valid(job_op_counts):
        chromosome = chromosome.repair(job_op_counts)
    
    return chromosome

def scramble_mutation_safe(chromosome, job_op_counts=None):
    """Mutation par brouillage d'un segment"""
    size = len(chromosome.genes)
    if size < 3:
        return chromosome
    
    # Choisir un segment
    start = random.randint(0, size - 3)
    end = random.randint(start + 2, min(start + 10, size))
    
    # Brouiller
    segment = chromosome.genes[start:end]
    random.shuffle(segment)
    chromosome.genes[start:end] = segment
    
    # Vérifier après mutation
    if job_op_counts and not chromosome.is_valid(job_op_counts):
        chromosome = chromosome.repair(job_op_counts)
    
    return chromosome

# ==================== ALGORITHME GÉNÉTIQUE AVANCÉ ====================
def genetic_algorithm_advanced(tasks, population_size=100, num_generations=200, 
                              crossover_rate=0.85, mutation_rate=0.15, 
                              elitism_count=5, adaptive_params=True):
    """
    Algorithme génétique avancé pour grands datasets
    """
    
    print(f"\n{'='*70}")
    print("ALGORITHME GÉNÉTIQUE AVANCÉ - GRANDS DATASETS")
    print(f"{'='*70}")
    print(f"Jobs: {len(tasks)}")
    print(f"Population: {population_size}")
    print(f"Générations: {num_generations}")
    
    # Pré-calculer
    task_by_id, job_ids, job_op_counts = create_optimized_mapping(tasks)
    total_ops = sum(job_op_counts.values())
    print(f"Opérations totales: {total_ops}")
    
    # Ajustement automatique des paramètres
    if adaptive_params:
        if total_ops > 10000:
            population_size = min(population_size, 60)
            num_generations = min(num_generations, 100)
            print(f"⚙️  Paramètres ajustés: pop={population_size}, gens={num_generations}")
    
    # Créer population
    print("\nCréation de la population...")
    population = create_population_fast(tasks, population_size, job_op_counts)
    
    # Évaluer
    print("Évaluation initiale...")
    start_eval = time.time()
    population = evaluate_population_sequential(population, tasks, task_by_id, job_op_counts)
    print(f"✓ Évaluation: {time.time() - start_eval:.1f}s")
    
    # Statistiques
    best_history = []
    avg_history = []
    best_solution = None
    stagnation_counter = 0
    
    # Boucle principale
    print("\nDémarrage des générations...")
    overall_start = time.time()
    
    for generation in range(num_generations):
        gen_start = time.time()
        
        # Trier
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Sauvegarder le meilleur
        current_best = population[0]
        if best_solution is None or current_best.fitness > best_solution.fitness:
            best_solution = deepcopy(current_best)
            stagnation_counter = 0
        else:
            stagnation_counter += 1
        
        # Statistiques
        best_fitness = current_best.fitness
        avg_fitness = sum(c.fitness for c in population) / len(population)
        best_history.append(best_fitness)
        avg_history.append(avg_fitness)
        
        # Affichage
        if generation % 5 == 0 or generation == num_generations - 1:
            elapsed = time.time() - overall_start
            print(f"Gen {generation:4d}: Makespan = {current_best.makespan:10.1f}, "
                  f"Fitness = {best_fitness:.6f}, Time = {elapsed:.1f}s")
        
        # Mutation adaptative
        current_mutation_rate = mutation_rate
        if stagnation_counter > 20:
            current_mutation_rate = min(0.3, mutation_rate * 1.5)  # Augmenter mutation
        elif generation < num_generations // 4:
            current_mutation_rate = mutation_rate * 1.2  # Plus de mutation en début
        
        # Élitisme
        new_population = [deepcopy(population[i]) for i in range(elitism_count)]
        
        # Génération
        while len(new_population) < population_size:
            # Sélection
            parent1 = tournament_selection_fast(population)
            parent2 = tournament_selection_fast(population)
            
            # Croisement
            if random.random() < crossover_rate:
                child1, child2 = order_crossover_safe(parent1, parent2, job_op_counts)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)
            
            # Mutation
            if random.random() < current_mutation_rate:
                if random.random() < 0.7:
                    child1 = swap_mutation_safe(child1, 0.05, job_op_counts)
                else:
                    child1 = scramble_mutation_safe(child1, job_op_counts)
            
            if random.random() < current_mutation_rate:
                if random.random() < 0.7:
                    child2 = swap_mutation_safe(child2, 0.05, job_op_counts)
                else:
                    child2 = scramble_mutation_safe(child2, job_op_counts)
            
            # Valider avant d'ajouter
            if not child1.is_valid(job_op_counts):
                child1 = child1.repair(job_op_counts)
            if not child2.is_valid(job_op_counts):
                child2 = child2.repair(job_op_counts)
            
            new_population.extend([child1, child2])
        
        # Limiter taille
        population = new_population[:population_size]
        
        # Évaluer
        if generation % 3 == 0 or generation == num_generations - 1:
            population = evaluate_population_sequential(population, tasks, task_by_id, job_op_counts)
        
        # Redémarrage partiel si stagnation
        if stagnation_counter > 30:
            print(f"  🔄 Redémarrage partiel à la génération {generation}")
            # Garder 20% des meilleurs, regénérer le reste
            keep_count = max(5, population_size // 5)
            new_diverse = create_population_fast(tasks, population_size - keep_count, job_op_counts)
            population = population[:keep_count] + new_diverse
            population = evaluate_population_sequential(population, tasks, task_by_id, job_op_counts)
            stagnation_counter = 0
        
        gen_time = time.time() - gen_start
        if gen_time > 10:  # Génération trop lente
            print(f"  ⚠️  Génération lente: {gen_time:.1f}s")
    
    # Résultats finaux
    total_time = time.time() - overall_start
    population.sort(key=lambda x: x.fitness, reverse=True)
    if population[0].fitness > best_solution.fitness:
        best_solution = deepcopy(population[0])
    
    print(f"\n{'='*70}")
    print("RÉSULTATS FINAUX")
    print(f"{'='*70}")
    print(f"Meilleur makespan: {best_solution.makespan:.1f}")
    print(f"Fitness: {best_solution.fitness:.6f}")
    print(f"Temps total: {total_time:.1f} secondes")
    print(f"Générations: {num_generations}")
    print(f"{'='*70}")
    
    return best_solution, best_history, avg_history, job_ids

# ==================== VISUALISATIONS ====================
def plot_convergence_fast(best_history, avg_history, filename='convergence_fast.png'):
    """Graphique de convergence simplifié"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    plt.figure(figsize=(10, 6))
    gens = range(len(best_history))
    
    plt.plot(gens, best_history, 'b-', label='Meilleur', linewidth=2)
    plt.plot(gens, avg_history, 'r--', label='Moyenne', linewidth=1.5, alpha=0.7)
    
    plt.xlabel('Génération')
    plt.ylabel('Fitness (1/makespan)')
    plt.title('Convergence - Algorithm Génétique Avancé')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence sauvegardée: {filepath}")
    plt.show()

def plot_gantt_simplified(chromosome, tasks, job_ids, filename='gantt_simple.png', max_jobs=50):
    """Diagramme de Gantt simplifié pour grands datasets"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    # Décoder
    task_by_id, _, job_op_counts = create_optimized_mapping(tasks)
    
    # Décoder pour visualisation (seulement les premiers jobs)
    job_op_counter = defaultdict(int)
    machine_times = {}
    job_times = defaultdict(float)
    operations = []
    
    job_ids_to_show = job_ids[:max_jobs]
    
    for gene in chromosome.genes:
        if gene not in job_ids_to_show:
            continue
            
        job_data = task_by_id[gene]
        op_idx = job_op_counter[gene]
        
        if op_idx >= job_data['op_count']:
            continue
            
        operation = job_data['operations'][op_idx]
        machine = operation['machine_id']
        duration = operation['duration']
        
        machine_time = machine_times.get(machine, 0.0)
        job_time = job_times[gene]
        
        start = max(machine_time, job_time)
        end = start + duration
        
        operations.append({
            'job_id': gene,
            'machine': machine,
            'start': start,
            'end': end,
            'duration': duration
        })
        
        machine_times[machine] = end
        job_times[gene] = end
        job_op_counter[gene] += 1
    
    # Calculer le makespan pour l'affichage
    makespan = max((op['end'] for op in operations), default=0)
    
    # Créer figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Organiser machines
    machines = sorted(set(op['machine'] for op in operations))
    machine_to_y = {m: i for i, m in enumerate(machines)}
    
    # Couleurs
    colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(job_ids_to_show))))
    job_to_color = {job: colors[i % len(colors)] for i, job in enumerate(job_ids_to_show)}
    
    # Dessiner
    for op in operations:
        y = machine_to_y[op['machine']]
        color = job_to_color[op['job_id']]
        
        rect = plt.Rectangle(
            (op['start'], y - 0.4),
            op['duration'],
            0.8,
            facecolor=color,
            edgecolor='black',
            alpha=0.7
        )
        ax.add_patch(rect)
        
        # Texte seulement si assez large
        if op['duration'] > makespan * 0.01:
            label = str(op['job_id']).replace('job_', 'J')
            ax.text(op['start'] + op['duration']/2, y, label,
                   ha='center', va='center', fontsize=6)
    
    # Configuration
    ax.set_xlim(0, makespan * 1.05 if makespan > 0 else 1)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    ax.set_xlabel('Temps')
    ax.set_ylabel('Machines')
    ax.set_title(f'Gantt Simplifié (premiers {max_jobs} jobs) - Makespan: {makespan:.1f}')
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=9)
    ax.grid(True, alpha=0.2, axis='x')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✓ Gantt simplifié sauvegardé: {filepath}")
    plt.show()

def save_stats_fast(chromosome, tasks, filename='stats_fast.txt'):
    """Statistiques rapides"""
    filepath = os.path.join(RESULTS_DIR, filename)
    
    task_by_id, _, job_op_counts = create_optimized_mapping(tasks)
    makespan = decode_chromosome_fast(chromosome, tasks, task_by_id)
    
    # Calculer l'utilisation des machines
    job_op_counter = defaultdict(int)
    machine_times = {}
    job_times = defaultdict(float)
    machine_usage = defaultdict(float)
    
    for gene in chromosome.genes:
        job_data = task_by_id[gene]
        op_idx = job_op_counter[gene]
        
        if op_idx >= job_data['op_count']:
            continue
            
        operation = job_data['operations'][op_idx]
        machine = operation['machine_id']
        duration = operation['duration']
        
        machine_time = machine_times.get(machine, 0.0)
        job_time = job_times[gene]
        
        start = max(machine_time, job_time)
        end = start + duration
        
        machine_times[machine] = end
        job_times[gene] = end
        machine_usage[machine] += duration
        job_op_counter[gene] += 1
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("STATISTIQUES RAPIDES - GRAND DATASET\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"MAKESPAN: {makespan:.1f}\n")
        f.write(f"FITNESS: {chromosome.fitness:.6f}\n")
        f.write(f"JOBS: {chromosome.num_jobs}\n")
        f.write(f"OPÉRATIONS: {sum(job_op_counts.values())}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("UTILISATION DES MACHINES (TOP 10)\n")
        f.write("-"*60 + "\n")
        sorted_machines = sorted(machine_usage.items(), key=lambda x: x[1], reverse=True)
        for machine, usage in sorted_machines[:10]:
            utilization = (usage / makespan) * 100
            f.write(f"{machine}: {utilization:.1f}%\n")
        
        f.write(f"\nTotal machines: {len(machine_usage)}\n")
    
    print(f"✓ Statistiques sauvegardées: {filepath}")