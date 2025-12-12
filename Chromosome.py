import json
import math
import random
import argparse
import time
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Dict, Tuple, Set
import tracemalloc
from multiprocessing import Pool, cpu_count
import numpy as np

tracemalloc.start()

# -------------------- GLOBAL CACHE --------------------
PREDS_CACHE = {}
GRAPH_CACHE = {} 
TASKS_DICT_GLOBAL = None
MACHINE_IDS_GLOBAL = None


# -------------------- IO HELPERS --------------------
#Charger la configuration des machines depuis JSON
def load_machines(machines_path: Path) -> List[str]:
    if not machines_path.exists():
        raise FileNotFoundError(f"machines.json not found at: {machines_path}")
    with machines_path.open("r", encoding="utf-8") as f:
        machines = json.load(f)
    machine_ids: List[str] = []
    if isinstance(machines, list):
        for m in machines:
            if isinstance(m, dict) and "id" in m:
                machine_ids.append(m["id"])
            elif isinstance(m, (int, str)):
                machine_ids.append(str(m))
    if not machine_ids:
        raise ValueError("No machine IDs found in machines.json.")
    return machine_ids

#Charger toutes les tâches avec leurs opérations et prédecesseurs
def load_tasks(tasks_path: Path) -> Dict[int, dict]:
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks file not found at: {tasks_path}")
    with tasks_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)
    task_dict: Dict[int, dict] = {}
    for t in tasks:
        task_dict[int(t["id"])] = t
    return task_dict


# -------------------- CACHE INITIALIZATION --------------------
#Précaculer TOUTES les relations de précédence une seule fois pour éviter de recalculer à chaque évaluation
def initialize_caches(tasks_dict: Dict[int, dict]):
    """Pre-compute all predecessor relationships once"""
    global PREDS_CACHE, GRAPH_CACHE
    
    PREDS_CACHE = {
        tid: set(int(p) for p in t.get("predecessors", []))
        for tid, t in tasks_dict.items()
    }
    
    # Build successor graph
    GRAPH_CACHE = defaultdict(list)
    for tid, preds in PREDS_CACHE.items():
        for p in preds:
            GRAPH_CACHE[p].append(tid)


# -------------------- OPTIMIZED REPAIR --------------------
# Obtenir un ordre valide de taches qui respecte toutes les contraintes, prêt à être évalué par le GA.
def precedence_respecting_repair_fast(order: List[int]) -> List[int]:
    """Optimized O(n) repair using cached predecessors"""
    placed: Set[int] = set()
    result: List[int] = []
    remaining = deque(order)
    
    # suivre combien de prédécesseurs restent à placer
    in_degree = {tid: len(PREDS_CACHE.get(tid, set())) for tid in order}
    
    while remaining:
        progressed = False
        for _ in range(len(remaining)):
            tid = remaining.popleft()
            
            if in_degree[tid] == 0:
                result.append(tid)
                placed.add(tid)
                progressed = True
                
                # Update in-degrees of successors
                for succ in GRAPH_CACHE.get(tid, []):
                    if succ not in placed:
                        in_degree[succ] -= 1
            else:
                remaining.append(tid)
        
        if not progressed and remaining:
            # Force placement to avoid infinite loop
            tid = remaining.popleft()
            result.append(tid)
            placed.add(tid)
            in_degree[tid] = 0
    
    return result


# -------------------- OPTIMIZED DECODER --------------------
#Simuler l'exécution des tâches dans l'ordre donné et calculer le temps total (makespan)
def decode_priority_schedule_fast(priority: List[int]) -> Tuple[Dict[int,int], Dict[int,int], List[dict], int]:
    """Fast decoder with minimal allocations"""
    machine_available_time: Dict[str, int] = {mid: 0 for mid in MACHINE_IDS_GLOBAL}
    task_end_times: Dict[int, int] = {}
    op_schedule: List[dict] = []

    for tid in priority:
        task = TASKS_DICT_GLOBAL[tid]
        
        # Calculate earliest start
        earliest_start = 0
        for pred in PREDS_CACHE.get(tid, set()):
            earliest_start = max(earliest_start, task_end_times.get(pred, 0))
        
        current_time = earliest_start
        
        # Schedule operations
        for idx, op in enumerate(task.get("operations", [])):
            mid = str(op["machine_id"])
            dur = int(op["duration"])
            
            start_time = max(current_time, machine_available_time[mid])
            end_time = start_time + dur
            
            machine_available_time[mid] = end_time
            current_time = end_time
            
            op_schedule.append({
                "task_id": tid,
                "op_index": idx,
                "machine_id": mid,
                "start": start_time,
                "end": end_time,
            })
#moment où chaque tâche se termine
        task_end_times[tid] = current_time

#max(task_end_times.values):temps total de fin de toutes les tâches,
    makespan = max(task_end_times.values()) if task_end_times else 0
    return {}, task_end_times, op_schedule, makespan


# -------------------- OPTIMIZED GA OPERATORS --------------------
def ppx_crossover_improved(parent_a: List[int], parent_b: List[int]) -> List[int]:
    """Improved PPX crossover with better precedence handling"""
    child: List[int] = []
    placed: Set[int] = set()
    
    # Use both parents as starting points
    tasks_set = set(parent_a) | set(parent_b)
    
    # Create priority queues from both parents
    queue_a = deque(parent_a)
    queue_b = deque(parent_b)
    
    max_attempts = len(tasks_set) * 3
    attempts = 0
    
    while len(placed) < len(tasks_set) and attempts < max_attempts:
        attempts += 1
        
        # Try from parent A first
        found = False
        while queue_a and not found:
            task = queue_a.popleft()
            if task not in placed:
                preds = PREDS_CACHE.get(task, set())
                if preds.issubset(placed):
                    child.append(task)
                    placed.add(task)
                    found = True
                else:
                    queue_a.append(task)  # Put back for later
                    # Try a limited number of times before moving on
                    if len(queue_a) > len(tasks_set) * 2:
                        break
        
        if not found and queue_b:
            # Try from parent B
            while queue_b and not found:
                task = queue_b.popleft()
                if task not in placed:
                    preds = PREDS_CACHE.get(task, set())
                    if preds.issubset(placed):
                        child.append(task)
                        placed.add(task)
                        found = True
                    else:
                        queue_b.append(task)
                        if len(queue_b) > len(tasks_set) * 2:
                            break
        
        if not found:
            # Find any available task
            for task in tasks_set:
                if task not in placed:
                    preds = PREDS_CACHE.get(task, set())
                    if preds.issubset(placed):
                        child.append(task)
                        placed.add(task)
                        found = True
                        break
    
    # À la fin, si certaines tâches restent encore non placées → elles sont ajoutées avec une heuristique.
    remaining = [t for t in tasks_set if t not in placed]
    if remaining:
        # Sort by number of remaining predecessors (heuristic)
        remaining.sort(key=lambda t: len(PREDS_CACHE.get(t, set()) - placed))
        child.extend(remaining)
        if len(remaining) > 5:  # Only warn if many tasks remain
            print(f"   ⚠️ Added {len(remaining)} remaining tasks (minimal)")
    
    return child
#permettre au GA d’évaluer plusieurs individus en parallèle sans recalculer les caches à chaque fois.
def mutate_swap_fast(order: List[int], mutation_rate: float = 0.2) -> List[int]:
    """Fast mutation without full repair if possible"""
    if random.random() >= mutation_rate or len(order) < 2:
        return order
    
    out = order[:]
    i, j = random.sample(range(len(out)), 2)
    
    # Check if swap would violate precedence
    ti, tj = out[i], out[j]
    
    # Simple check: if neither is predecessor of the other, swap is safe
    if tj not in PREDS_CACHE.get(ti, set()) and ti not in PREDS_CACHE.get(tj, set()):
        out[i], out[j] = out[j], out[i]
        return out
    
    # Otherwise do swap and repair
    out[i], out[j] = out[j], out[i]
    
    # Si le swap risque de violer la précédence, swap anyway then repare order with precedence_respecting_repair_fast
    try:
        return precedence_respecting_repair_fast(out)
    except Exception as e:
        print(f"   ⚠️ Repair failed: {e}, returning original order")
        return order
    
def validate_schedule_order(order: List[int]) -> bool:
    """Validate that an order respects precedence constraints"""
    placed = set()
    for task in order:
        preds = PREDS_CACHE.get(task, set())
        if not preds.issubset(placed):
            print(f"   ❌ Invalid order: Task {task} has predecessors {preds} not in {placed}")
            return False
        placed.add(task)
    return True


# -------------------- PARALLEL EVALUATION --------------------
#→ Sert de wrapper pour ne récupérer que le makespan.
#→ Cela permet de noter la qualité (fitness) d’un individu dans le GA.
def eval_order_parallel(order: List[int]) -> int:
    """Wrapper for parallel evaluation"""
    _, _, _, ms = decode_priority_schedule_fast(order)
    return ms

#permettre au GA d’évaluer plusieurs individus en parallèle sans recalculer les caches à chaque fois.
def init_worker(tasks_dict, machine_ids):
    """Initialize global variables in each worker process"""
    global TASKS_DICT_GLOBAL, MACHINE_IDS_GLOBAL
    TASKS_DICT_GLOBAL = tasks_dict
    MACHINE_IDS_GLOBAL = machine_ids
    initialize_caches(tasks_dict)


# -------------------- ADAPTIVE GA --------------------
#- tasks_dict: données des tâches
#- machine_ids: liste des machines
#- seed: graine aléatoire pour reproductibilité (42 par défaut)
#- pop_size: taille population (50 par défaut)
#- generations: nombre max de générations (200 par défaut)
#- cx_rate: taux de crossover (0.9 = 90% par défaut)
#- mut_rate: taux de mutation (0.2 = 20% par défaut)
#- use_parallel: utiliser parallélisme (True par défaut)
#- early_stop_patience: arrêt si pas amélioration pendant X générations (30 par défaut)
#Boucle sur les générations:
#   . Élitisme (garder top 10%)
#   . Sélection par tournoi (choisir 3, prendre meilleur)
#   . Crossover (90% de chance)
#   . Mutation (taux adaptatif)
#   . Évaluation nouvelle population
#   . Mise à jour meilleure solution
#   . Arrêt prématuré si stagnation

def run_adaptive_ga(tasks_dict: Dict[int, dict], machine_ids: List[str],
                    seed: int | None = None,
                    pop_size: int = 50, 
                    generations: int = 200,
                    cx_rate: float = 0.9, 
                    mut_rate: float = 0.2,
                    use_parallel: bool = True,
                    early_stop_patience: int = 30) -> Tuple[Dict[int,int], Dict[int,int], List[dict], int, List[int]]:
    
    global TASKS_DICT_GLOBAL, MACHINE_IDS_GLOBAL
    TASKS_DICT_GLOBAL = tasks_dict
    MACHINE_IDS_GLOBAL = machine_ids
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    initialize_caches(tasks_dict)
    task_ids = list(tasks_dict.keys())
    
    print(f"🚀 Starting Adaptive GA: {len(task_ids)} tasks, {len(machine_ids)} machines")
    print(f"   Pop size: {pop_size}, Generations: {generations}, Parallel: {use_parallel}")

    # Initialize population with diverse topological sorts
    def generate_initial():
        shuffled = task_ids[:]
        random.shuffle(shuffled)
        return precedence_respecting_repair_fast(shuffled)
    
    population: List[List[int]] = [generate_initial() for _ in range(pop_size)]

    # Parallel pool setup - only use for large datasets
    pool = None
    should_use_parallel = use_parallel and len(task_ids) > 500
    if should_use_parallel:
        num_workers = max(1, min(4, cpu_count() - 1))  # Cap at 4 workers
        pool = Pool(processes=num_workers, initializer=init_worker, initargs=(tasks_dict, machine_ids))
        print(f"   Using {num_workers} parallel workers")
    else:
        if use_parallel:
            print(f"   Parallel disabled (< 500 tasks, overhead too high)")

    def eval_population(pop: List[List[int]]) -> List[int]:
        if should_use_parallel and pool:
            return pool.map(eval_order_parallel, pop, chunksize=max(1, len(pop)//num_workers))
        else:
            return [eval_order_parallel(order) for order in pop]

    # Initial evaluation
    fitnesses = eval_population(population)
    scored: List[Tuple[List[int], int]] = list(zip(population, fitnesses))
    scored.sort(key=lambda x: x[1])
    
    best_order, best_makespan = scored[0]
    print(f"   Initial best makespan: {best_makespan} ({best_makespan/60:.1f}h)")

    # Evolution with adaptive parameters and early stopping
    no_improvement_count = 0
    best_history = [best_makespan]
    
    start_time = time.perf_counter()
    
    for gen in range(generations):
        new_pop: List[List[int]] = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_pop.append(scored[i][0])
        
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            p1 = min(random.sample(scored, 3), key=lambda x: x[1])[0]
            p2 = min(random.sample(scored, 3), key=lambda x: x[1])[0]
            
            # Crossover
            if random.random() < cx_rate:
                child = ppx_crossover_improved(p1, p2)
            else:
                child = p1[:]
            
            # Adaptive mutation rate
            adaptive_mut = mut_rate * (1.5 if no_improvement_count > 10 else 1.0)
            child = mutate_swap_fast(child, mutation_rate=adaptive_mut)
            if len(child) != len(task_ids):
                print(f"   ⚠️ Child length mismatch: {len(child)} vs {len(task_ids)}, regenerating")
                child = generate_initial()

            new_pop.append(child)
        
        # Evaluate new population
        fitnesses = eval_population(new_pop)
        scored = list(zip(new_pop, fitnesses))
        scored.sort(key=lambda x: x[1])
        
        current_best_makespan = scored[0][1]
        
        # Update best solution / Stagnation: nbr générations consécutives où aucune amélioration done.
        if current_best_makespan < best_makespan:
            best_makespan = current_best_makespan
            best_order = scored[0][0]
            no_improvement_count = 0
            print(f"   Gen {gen+1}: NEW BEST = {best_makespan} ({best_makespan/60:.1f}h) ⭐")
        else:
            no_improvement_count += 1
        
        best_history.append(best_makespan)
        
        # Progress report every 10 generations for small datasets, 20 for large
        report_interval = 10 if len(task_ids) < 1000 else 20
        if (gen + 1) % report_interval == 0:
            elapsed = time.perf_counter() - start_time
            print(f"   Gen {gen+1}/{generations}: Best={best_makespan} ({best_makespan/60:.1f}h), "
                  f"Avg={np.mean(fitnesses):.0f}, No improvement={no_improvement_count}, "
                  f"Time={elapsed:.1f}s")
        
        # GA stops quand il n'apprend plus
        if no_improvement_count >= early_stop_patience:
            print(f"   🛑 Early stopping at generation {gen+1} (no improvement for {early_stop_patience} gens)")
            break
#gagner du temps et éviter de continuer inutilement quand le GA a convergé.
    if pool:
        pool.close()
        pool.join()

    # Final decode
    starts, ends, ops, _ = decode_priority_schedule_fast(best_order)
    
    total_time = time.perf_counter() - start_time
    print(f"✅ GA Complete: Best makespan = {best_makespan} ({best_makespan/60:.2f}h) in {total_time:.1f}s")
    
    return starts, ends, ops, best_makespan, best_order


# -------------------- VISUALIZATION --------------------
def visualize_gantt_optimized(op_schedule: List[dict], machine_order: List[str] | None = None, 
                             out_path: Path = Path("gantt_ga.png"), title: str = "GA Schedule",
                             sample_rate: int = 1) -> None:
    """Optimized Gantt chart with sampling for large schedules"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed; skipping Gantt chart.")
        return

    # Sample operations if too many
    if len(op_schedule) > 10000:
        print(f"   Sampling {len(op_schedule)} operations (showing every {sample_rate}th operation)")
        op_schedule = op_schedule[::sample_rate]

    ops_machines = [op["machine_id"] for op in op_schedule]
    machines = list(machine_order or [])
    seen = set(machines)
    for m in ops_machines:
        if m not in seen:
            machines.append(m)
            seen.add(m)

    y_index = {m: i for i, m in enumerate(machines)}
    time_max = max((op["end"] for op in op_schedule), default=0)

    fig_w = max(12, min(20, time_max / 100))
    fig_h = max(6, min(16, len(machines) * 0.4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Use a simpler color scheme for performance
    num_tasks = max((op["task_id"] for op in op_schedule), default=0) + 1
    colors = plt.cm.tab20(np.linspace(0, 1, 20))

    for op in op_schedule:
        mid = op["machine_id"]
        tid = op["task_id"]
        start = op["start"]
        end = op["end"]
        dur = end - start
        y = y_index[mid]
        
        color = colors[tid % 20]
        ax.broken_barh([(start, dur)], (y - 0.4, 0.8), 
                      facecolors=color, edgecolor="black", 
                      linewidth=0.5, alpha=0.85)
        
        # Only label if bar is wide enough
        if dur > time_max * 0.01:
            ax.text(start + dur / 2, y, f"T{tid}", 
                   va="center", ha="center", color="white", 
                   fontsize=6, clip_on=True, fontweight='bold')

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontsize=8)
    ax.set_xlabel("Time (minutes)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(1, time_max))
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"   💾 Saved Gantt chart to {out_path}")


# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized GA scheduler for large datasets")
    parser.add_argument("--machines", type=Path, default=Path("machines.json"))
    parser.add_argument("--tasks", type=Path, default=Path("tasks_small.json"))
    parser.add_argument("--out", type=Path, default=Path("gantt_ga_optimized.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pop-size", type=int, default=50, help="Population size (50-100 for large datasets)")
    parser.add_argument("--generations", type=int, default=200, help="Max generations (200-500 for large datasets)")
    parser.add_argument("--cx-rate", type=float, default=0.9)
    parser.add_argument("--mut-rate", type=float, default=0.2)
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--early-stop", type=int, default=30, help="Early stopping patience")
    args = parser.parse_args()

    print("="*60)
    print("🔧 OPTIMIZED GA SCHEDULER FOR LARGE DATASETS")
    print("="*60)

    machine_ids = load_machines(args.machines)
    tasks = load_tasks(args.tasks)
    
    print(f"📊 Loaded: {len(tasks)} tasks, {len(machine_ids)} machines")

    t0 = time.perf_counter()
    starts, ends, ops, ms, order = run_adaptive_ga(
        tasks, machine_ids,
        seed=args.seed,
        pop_size=args.pop_size,
        generations=args.generations,
        cx_rate=args.cx_rate,
        mut_rate=args.mut_rate,
        use_parallel=not args.no_parallel,
        early_stop_patience=args.early_stop
    )
    dt = time.perf_counter() - t0
    
    print("="*60)
    print(f"📈 FINAL RESULTS:")
    print(f"   Makespan: {ms} minutes ({ms/60:.2f} hours)")
    print(f"   Runtime: {dt:.2f} seconds")
    print(f"   Operations: {len(ops)}")
    print("="*60)
    
    # Save results to JSON
    results_path = Path("results_optimized.json")
    with results_path.open("w") as f:
        json.dump({
            "makespan_minutes": ms,
            "makespan_hours": ms/60,
            "runtime_seconds": dt,
            "num_tasks": len(tasks),
            "num_operations": len(ops),
            "priority_order": order,
            "parameters": {
                "pop_size": args.pop_size,
                "generations": args.generations,
                "cx_rate": args.cx_rate,
                "mut_rate": args.mut_rate,
                "seed": args.seed
            }
        }, f, indent=2)
    print(f"   💾 Saved results to {results_path}")
    
    # Visualize (with sampling if needed)
    sample_rate = max(1, len(ops) // 5000)
    visualize_gantt_optimized(
        ops,
        machine_order=machine_ids,
        out_path=args.out,
        title=f"Optimized GA Schedule ({len(tasks)} tasks, {ms/60:.1f}h, {dt:.0f}s)",
        sample_rate=sample_rate
    )
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"   💾 Memory: Current={current/1_000_000:.1f}MB, Peak={peak/1_000_000:.1f}MB")
    print("="*60)
    tracemalloc.stop()


if __name__ == "__main__":
    main()