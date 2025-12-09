import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
import numpy as np

# ==================== CHARGEMENT DES DONNÉES ====================
def load_tasks(filename='tasks.json'):
    """Charge les tâches depuis le fichier JSON"""
    with open(filename, 'r') as f:
        return json.load(f)

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
        # Compter les occurrences de chaque job
        expected_counts = {task['id']: len(task['operations']) for task in tasks}
        actual_counts = {}
        
        for gene in self.genes:
            if gene is None:
                return False, "Contient des gènes None"
            actual_counts[gene] = actual_counts.get(gene, 0) + 1
        
        # Vérifier que chaque job a le bon nombre d'occurrences
        for job_id, expected in expected_counts.items():
            if actual_counts.get(job_id, 0) != expected:
                return False, f"Job {job_id}: {actual_counts.get(job_id, 0)} au lieu de {expected}"
        
        return True, "Chromosome valide"

# ==================== INITIALISATION ====================
def create_initial_population(tasks, population_size):
    """Crée la population initiale avec des permutations aléatoires"""
    population = []
    
    # Créer la liste de toutes les opérations
    all_operations = []
    for task in tasks:
        task_id = task['id']
        num_operations = len(task['operations'])
        for _ in range(num_operations):
            all_operations.append(task_id)
    
    print(f"Nombre total d'opérations: {len(all_operations)}")
    
    # Vérifier que chaque job a le bon nombre d'occurrences
    expected_counts = {task['id']: len(task['operations']) for task in tasks}
    actual_counts = {}
    for op in all_operations:
        actual_counts[op] = actual_counts.get(op, 0) + 1
    
    for job_id, expected in expected_counts.items():
        actual = actual_counts.get(job_id, 0)
        if actual != expected:
            print(f"ATTENTION: Job {job_id} a {actual} occurrences au lieu de {expected}")
    
    # Générer population_size permutations aléatoires
    for i in range(population_size):
        genes = all_operations.copy()
        random.shuffle(genes)
        chromosome = Chromosome(genes, len(tasks))
        
        # Vérifier la validité
        is_valid, msg = chromosome.is_valid(tasks)
        if not is_valid:
            print(f"ERREUR: Chromosome initial invalide: {msg}")
            # Réparer le chromosome
            chromosome = repair_chromosome(chromosome, tasks)
        
        population.append(chromosome)
    
    return population

# ==================== FONCTIONS DE RÉPARATION ====================
def repair_chromosome(chromosome, tasks):
    """Répare un chromosome invalide"""
    # Compter les occurrences actuelles
    actual_counts = {}
    for gene in chromosome.genes:
        if gene is not None:
            actual_counts[gene] = actual_counts.get(gene, 0) + 1
    
    # Déterminer les occurrences requises
    required_counts = {task['id']: len(task['operations']) for task in tasks}
    
    # Liste des gènes manquants
    missing_genes = []
    for job_id, required in required_counts.items():
        actual = actual_counts.get(job_id, 0)
        if actual < required:
            # Ajouter les gènes manquants
            missing_genes.extend([job_id] * (required - actual))
    
    # Remplacer les gènes None par les gènes manquants
    repaired_genes = []
    missing_idx = 0
    
    for gene in chromosome.genes:
        if gene is None and missing_idx < len(missing_genes):
            repaired_genes.append(missing_genes[missing_idx])
            missing_idx += 1
        elif gene is not None:
            repaired_genes.append(gene)
    
    # S'il reste des gènes manquants, les ajouter à la fin
    while missing_idx < len(missing_genes):
        repaired_genes.append(missing_genes[missing_idx])
        missing_idx += 1
    
    return Chromosome(repaired_genes, chromosome.num_jobs)

# ==================== DÉCODAGE ET ÉVALUATION ====================
def decode_chromosome(chromosome, tasks, task_by_id=None):
    """Décode le chromosome en ordonnancement et calcule le makespan"""
    # Créer le mapping si non fourni
    if task_by_id is None:
        task_by_id, _ = create_task_mapping(tasks)
    
    # Compteurs pour suivre quelle opération de chaque job
    job_operation_counter = {}
    for task in tasks:
        job_operation_counter[task['id']] = 0
    
    # Temps de fin pour chaque machine et chaque job
    machine_end_times = {}
    job_end_times = {}
    for task in tasks:
        job_end_times[task['id']] = 0
    
    # Ordonnancement complet
    schedule = []
    
    # Parcourir chaque gène (job_id)
    for gene in chromosome.genes:
        # Vérifier que le gène n'est pas None
        if gene is None:
            print(f"ATTENTION: Gène None trouvé dans le chromosome!")
            # Utiliser un job par défaut (le premier)
            gene = tasks[0]['id']
        
        job_id = gene
        operation_index = job_operation_counter[job_id]
        
        # Récupérer l'opération via le dictionnaire
        task = task_by_id[job_id]
        operation = task['operations'][operation_index]
        machine_id = operation['machine_id']
        duration = operation['duration']
        
        # Initialiser la machine si nécessaire
        if machine_id not in machine_end_times:
            machine_end_times[machine_id] = 0
        
        # L'opération commence quand la machine ET le job sont libres
        start_time = max(machine_end_times[machine_id], job_end_times[job_id])
        end_time = start_time + duration
        
        # Mettre à jour les temps
        machine_end_times[machine_id] = end_time
        job_end_times[job_id] = end_time
        
        # Enregistrer dans l'ordonnancement
        schedule.append({
            'job_id': job_id,
            'operation_index': operation_index,
            'machine_id': machine_id,
            'start': start_time,
            'end': end_time,
            'duration': duration
        })
        
        # Incrémenter le compteur d'opérations pour ce job
        job_operation_counter[job_id] += 1
    
    # Le makespan est le temps de fin maximum
    makespan = max(job_end_times.values())
    
    return schedule, makespan

def evaluate_fitness(chromosome, tasks, task_by_id=None):
    """Évalue le fitness d'un chromosome"""
    # Vérifier la validité d'abord
    is_valid, msg = chromosome.is_valid(tasks)
    if not is_valid:
        # Réparer le chromosome
        chromosome = repair_chromosome(chromosome, tasks)
    
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    chromosome.makespan = makespan
    # Fitness inversement proportionnel au makespan (minimisation)
    chromosome.fitness = 1.0 / makespan if makespan > 0 else 0
    return chromosome.fitness

# ==================== SÉLECTION ====================
def tournament_selection(population, tournament_size=3):
    """Sélection par tournoi"""
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda x: x.fitness)

# ==================== CROISEMENT ====================
def order_crossover(parent1, parent2):
    """Order Crossover (OX) - robuste pour JSSP"""
    size = len(parent1.genes)
    
    # Choisir deux points de croisement
    point1 = random.randint(0, size - 2)
    point2 = random.randint(point1 + 1, size)
    
    # Initialiser enfants
    child1 = [None] * size
    child2 = [None] * size
    
    # Copier le segment entre point1 et point2
    for i in range(point1, point2):
        child1[i] = parent1.genes[i]
        child2[i] = parent2.genes[i]
    
    # Remplir child1 avec les gènes de parent2 (en évitant les doublons)
    pos = point2
    for i in range(size):
        idx = (point2 + i) % size
        gene = parent2.genes[idx]
        if gene not in child1:
            child1[pos % size] = gene
            pos += 1
            if pos >= size:
                pos = 0
    
    # Remplir child2 avec les gènes de parent1 (en évitant les doublons)
    pos = point2
    for i in range(size):
        idx = (point2 + i) % size
        gene = parent1.genes[idx]
        if gene not in child2:
            child2[pos % size] = gene
            pos += 1
            if pos >= size:
                pos = 0
    
    # S'assurer qu'il n'y a pas de None
    if None in child1:
        # Compléter avec les gènes manquants
        all_genes = list(set(parent1.genes + parent2.genes))
        missing_genes = [g for g in all_genes if g not in child1]
        for i in range(size):
            if child1[i] is None and missing_genes:
                child1[i] = missing_genes.pop(0)
    
    if None in child2:
        # Compléter avec les gènes manquants
        all_genes = list(set(parent1.genes + parent2.genes))
        missing_genes = [g for g in all_genes if g not in child2]
        for i in range(size):
            if child2[i] is None and missing_genes:
                child2[i] = missing_genes.pop(0)
    
    return Chromosome(child1, parent1.num_jobs), Chromosome(child2, parent2.num_jobs)

# ==================== MUTATION ====================
def swap_mutation(chromosome):
    """Mutation par échange de deux gènes"""
    size = len(chromosome.genes)
    if size < 2:
        return chromosome
    
    pos1 = random.randint(0, size - 1)
    pos2 = random.randint(0, size - 1)
    
    # Échanger (s'assurer qu'on a deux positions différentes)
    while pos2 == pos1:
        pos2 = random.randint(0, size - 1)
    
    chromosome.genes[pos1], chromosome.genes[pos2] = \
        chromosome.genes[pos2], chromosome.genes[pos1]
    
    return chromosome

# ==================== ALGORITHME GÉNÉTIQUE PRINCIPAL ====================
def genetic_algorithm(tasks, population_size=50, crossover_rate=0.8, 
                     mutation_rate=0.1, num_generations=100, elitism_count=2):
    """
    Algorithme génétique pour résoudre le JSSP
    """
    
    print(f"=== Démarrage de l'Algorithme Génétique ===")
    print(f"Nombre de jobs: {len(tasks)}")
    print(f"Taille population: {population_size}")
    print(f"Taux croisement: {crossover_rate}")
    print(f"Taux mutation: {mutation_rate}")
    print(f"Générations: {num_generations}")
    print(f"Élitisme: {elitism_count}")
    
    # Calculer le nombre total d'opérations
    total_operations = sum(len(task['operations']) for task in tasks)
    print(f"Total opérations: {total_operations}")
    
    # Créer le mapping une fois pour toutes
    task_by_id, job_ids = create_task_mapping(tasks)
    
    # Initialisation
    print("\nCréation de la population initiale...")
    population = create_initial_population(tasks, population_size)
    
    # Évaluer la population initiale
    print("Évaluation de la population initiale...")
    for individual in population:
        evaluate_fitness(individual, tasks, task_by_id)
    
    # Statistiques
    best_fitness_history = []
    avg_fitness_history = []
    best_solution = None
    
    # Boucle principale
    print("\nDémarrage des générations...")
    for generation in range(num_generations):
        # Trier par fitness
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Sauvegarder le meilleur
        if best_solution is None or population[0].fitness > best_solution.fitness:
            best_solution = deepcopy(population[0])
        
        # Statistiques
        best_fitness = population[0].fitness
        avg_fitness = sum(ind.fitness for ind in population) / len(population)
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Affichage de progression
        if generation % 10 == 0 or generation == num_generations - 1:
            print(f"Génération {generation:3d}: Makespan = {population[0].makespan:8.2f}, "
                  f"Fitness = {best_fitness:.6f}")
        
        # Élitisme: conserver les meilleurs
        new_population = deepcopy(population[:elitism_count])
        
        # Générer le reste de la population
        while len(new_population) < population_size:
            # Sélection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Croisement
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
            
            # Vérifier et réparer si nécessaire
            is_valid1, msg1 = child1.is_valid(tasks)
            if not is_valid1:
                child1 = repair_chromosome(child1, tasks)
            
            is_valid2, msg2 = child2.is_valid(tasks)
            if not is_valid2:
                child2 = repair_chromosome(child2, tasks)
            
            # Ajouter les enfants s'il y a de la place
            if len(new_population) < population_size:
                new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        # Remplacer la population
        population = new_population[:population_size]
        
        # Évaluer la nouvelle population
        for individual in population:
            evaluate_fitness(individual, tasks, task_by_id)
    
    # Résultats finaux
    population.sort(key=lambda x: x.fitness, reverse=True)
    if population[0].fitness > best_solution.fitness:
        best_solution = deepcopy(population[0])
    
    print(f"\n{'='*50}")
    print(f"=== RÉSULTATS FINAUX ===")
    print(f"{'='*50}")
    print(f"Meilleur makespan trouvé: {best_solution.makespan:.2f}")
    print(f"Fitness: {best_solution.fitness:.6f}")
    print(f"Nombre de générations: {num_generations}")
    print(f"{'='*50}")
    
    return best_solution, best_fitness_history, avg_fitness_history, job_ids

# ==================== VISUALISATION - DIAGRAMME DE GANTT ====================
def plot_gantt_chart(chromosome, tasks, job_ids, filename='gantt_chart.png'):
    """Génère un diagramme de Gantt pour la solution"""
    task_by_id, _ = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    # Créer une figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Créer un mapping job_id -> index pour les couleurs
    job_to_index = {job_id: idx for idx, job_id in enumerate(job_ids)}
    
    # Couleurs pour chaque job
    num_jobs = len(job_ids)
    if num_jobs <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_jobs))
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, num_jobs))
    
    # Obtenir toutes les machines uniques et les trier
    machines = sorted(set(op['machine_id'] for op in schedule))
    machine_to_y = {machine: i for i, machine in enumerate(machines)}
    
    print(f"\nGénération du diagramme de Gantt...")
    print(f"Nombre de machines: {len(machines)}")
    print(f"Makespan: {makespan}")
    print(f"Nombre d'opérations: {len(schedule)}")
    
    # Dessiner chaque opération
    for operation in schedule:
        job_id = operation['job_id']
        job_index = job_to_index[job_id]
        machine_id = operation['machine_id']
        start = operation['start']
        duration = operation['duration']
        
        y_pos = machine_to_y[machine_id]
        
        # Créer un rectangle
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
        
        # Ajouter le texte (Job ID) si la durée est suffisante
        if duration > makespan * 0.02:  # Afficher seulement si assez large
            label = str(job_id).replace('job_', 'J')
            ax.text(start + duration/2, y_pos, label,
                    ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Configuration des axes
    ax.set_xlim(0, makespan * 1.01)
    ax.set_ylim(-0.5, len(machines) - 0.5)
    ax.set_xlabel('Temps', fontsize=14, fontweight='bold')
    ax.set_ylabel('Machines', fontsize=14, fontweight='bold')
    ax.set_title(f'Diagramme de Gantt - Job Shop Scheduling\nMakespan: {makespan:.2f}', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grille
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([str(m) for m in machines], fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.grid(axis='y', alpha=0.1, linestyle='-')
    
    # Légende (afficher seulement les 15 premiers jobs)
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
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Diagramme de Gantt sauvegardé: {filename}")
    plt.show()

def plot_convergence(best_fitness_history, avg_fitness_history, 
                     filename='convergence.png'):
    """Trace la convergence de l'algorithme"""
    plt.figure(figsize=(12, 7))
    
    generations = range(len(best_fitness_history))
    
    # Tracer les courbes
    plt.plot(generations, best_fitness_history, 'b-', 
             label='Meilleur Fitness', linewidth=2.5, alpha=0.8)
    plt.plot(generations, avg_fitness_history, 'r--', 
             label='Fitness Moyen', linewidth=2, alpha=0.7)
    
    # Configuration
    plt.xlabel('Génération', fontsize=13, fontweight='bold')
    plt.ylabel('Fitness (1/Makespan)', fontsize=13, fontweight='bold')
    plt.title('Convergence de l\'Algorithme Génétique - JSSP', 
              fontsize=15, fontweight='bold', pad=15)
    
    # Légende et grille
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Graphique de convergence sauvegardé: {filename}")
    plt.show()

def save_solution_stats(chromosome, tasks, filename='solution_stats.txt'):
    """Sauvegarde les statistiques de la solution"""
    task_by_id, _ = create_task_mapping(tasks)
    schedule, makespan = decode_chromosome(chromosome, tasks, task_by_id)
    
    # Calculer les statistiques
    machine_utilization = {}
    job_completion_times = {}
    
    for op in schedule:
        machine_id = op['machine_id']
        job_id = op['job_id']
        duration = op['duration']
        
        # Utilisation des machines
        if machine_id not in machine_utilization:
            machine_utilization[machine_id] = 0
        machine_utilization[machine_id] += duration
        
        # Temps de fin des jobs
        job_completion_times[job_id] = max(job_completion_times.get(job_id, 0), op['end'])
    
    # Sauvegarder
    with open(filename, 'w') as f:
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
        f.write("TEMPS DE FIN DES JOBS\n")
        f.write("-"*60 + "\n")
        for job_id in sorted(job_completion_times.keys())[:20]:  # Limiter aux 20 premiers
            f.write(f"{job_id}: {job_completion_times[job_id]:.2f}\n")
        if len(job_completion_times) > 20:
            f.write(f"... et {len(job_completion_times) - 20} autres jobs\n")
    
    print(f"✓ Statistiques sauvegardées: {filename}")