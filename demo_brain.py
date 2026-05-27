"""
===========================================================
PRECIOUS BRAIN - WORKING DEMO
-----------------------------------------------------------
A practical demonstration of the unified cognitive
architecture with all systems working together.

Run:  python demo_brain.py
===========================================================
"""

import numpy as np
import time
import sys

# Import the brain
from A_Brain import (
    PreciousBrain,
    PreciousBrainConfig,
    create_precious_brain,
    CognitiveMode
)

from Neural_Consciousness import EmotionType


def separator(title, emoji="🧠"):
    """Pretty section separator"""
    print(f"\n{'━'*60}")
    print(f"  {emoji} {title}")
    print(f"{'━'*60}\n")


# ============================================================
# DEMO 1: Create the Brain
# ============================================================
def demo_create_brain():
    """Create and initialize the brain"""
    separator("STEP 1: CREATING THE BRAIN", "🔧")

    brain = create_precious_brain(
        name="MyBrain",
        architecture="medium",
        input_size=1,       # 1D input for simplicity
        output_size=1,      # 1D output
        n_consciousness_neurons=3000,  # Smaller for faster demo
        population_size=4,             # Smaller population
        evolution_generations=3,       # Fewer generations
        enable_all=True
    )

    print("✅ Brain created successfully!")
    return brain


# ============================================================
# DEMO 2: Think, Feel, Decide
# ============================================================
def demo_cognitive_ops(brain):
    """Demonstrate basic cognitive operations"""
    separator("STEP 2: COGNITIVE OPERATIONS", "💭")

    # --- Thinking ---
    print("💭 Forming a thought...")
    input_signal = np.array([0.75])  # A stimulus
    thought = brain.think(input_signal, context={'name': 'first_stimulus'})
    print(f"   Neurons activated: {len(thought.active_neurons)}")
    print(f"   Activation strength: {thought.activation_strength:.4f}")
    print(f"   Pattern ID: {thought.pattern_id}")

    # --- Feeling ---
    print("\n💗 Processing emotions...")

    # Calm state
    emotion_calm = brain.feel(
        np.array([0.2]),
        body_signals={'heart_rate': 65, 'arousal': 0.1}
    )
    print(f"   Calm input → Emotion: {emotion_calm.emotion.value} "
          f"(intensity: {emotion_calm.intensity:.3f})")

    # Excited state
    emotion_excited = brain.feel(
        np.array([0.9]),
        body_signals={'heart_rate': 110, 'arousal': 0.8}
    )
    print(f"   Excited input → Emotion: {emotion_excited.emotion.value} "
          f"(intensity: {emotion_excited.intensity:.3f})")

    # --- Decision Making ---
    print("\n🎯 Making a pre-conscious decision...")
    decision_time, label, conscious = brain.decide(
        np.array([0.6]),
        decision_label="should_I_act"
    )
    print(f"   Decision made at: {decision_time:.6f}")
    print(f"   Label: {label}")
    print(f"   Conscious delay: {brain.config.decision_latency_ms}ms")

    # Wait for consciousness to catch up
    time.sleep(0.35)
    conscious_now = brain.consciousness.check_conscious_awareness()
    if conscious_now:
        print(f"   ✅ Now conscious of: {conscious_now}")
    else:
        print(f"   ⏳ Still pre-conscious (need {brain.config.decision_latency_ms}ms)")


# ============================================================
# DEMO 3: Memory Operations
# ============================================================
def demo_memory(brain):
    """Demonstrate memory storage and recall"""
    separator("STEP 3: MEMORY OPERATIONS", "💾")

    # Store several experiences
    print("📥 Storing memories...")
    experiences = [
        (np.array([1.0]), np.array([0.84]), EmotionType.HAPPINESS, "sine_peak"),
        (np.array([0.0]), np.array([0.0]),  EmotionType.NEUTRAL,   "zero_point"),
        (np.array([-1.0]), np.array([-0.84]), EmotionType.SADNESS, "sine_valley"),
        (np.array([0.5]), np.array([0.48]), EmotionType.HAPPINESS, "rising_curve"),
    ]

    for data, label, emotion, name in experiences:
        brain.memorize(data, label, emotional_context=emotion)
        print(f"   ✅ Stored '{name}': input={data[0]:.1f}, "
              f"output={label[0]:.2f}, emotion={emotion.value}")

    # Recall
    print("\n🔍 Recalling memories...")
    query = np.array([0.9])
    from Hippocampus_Brain import MemoryType
    memories = brain.recall(query, memory_type=MemoryType.SHORT_TERM)
    print(f"   Query: {query[0]:.1f}")
    print(f"   Retrieved: {len(memories)} memories")
    for i, mem in enumerate(memories[:3]):
        print(f"   [{i+1}] content={mem.content.flatten()[0]:.2f}, "
              f"importance={mem.importance:.3f}")


# ============================================================
# DEMO 4: Learning from Data
# ============================================================
def demo_learning(brain):
    """Train the brain on a real function"""
    separator("STEP 4: LEARNING A FUNCTION", "📈")

    # Generate training data: y = sin(x)
    X_train = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_train = np.sin(X_train)

    print("📊 Training data: y = sin(x)")
    print(f"   Samples: {len(X_train)}")
    print(f"   X range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"   y range: [{y_train.min():.2f}, {y_train.max():.2f}]\n")

    # Learn!
    results = brain.learn(
        X_train, y_train,
        epochs=60,
        use_evolution=True,
        use_memory_replay=True,
        consolidate_freq=10,
        verbose=True
    )

    return X_train, y_train


# ============================================================
# DEMO 5: Prediction & Evaluation
# ============================================================
def demo_prediction(brain, X_train, y_train):
    """Test the trained brain"""
    separator("STEP 5: PREDICTIONS", "🔮")

    # Predict on training data
    predictions = brain.predict(X_train)
    train_error = np.mean(np.abs(predictions - y_train))
    print(f"📊 Training Error (MAE): {train_error:.6f}")

    # Predict on NEW data (generalization)
    X_test = np.array([[-2.5], [-1.0], [0.0], [1.0], [2.5]])
    y_true = np.sin(X_test)
    y_pred = brain.predict(X_test)

    print(f"\n🧪 Generalization Test:")
    print(f"   {'x':>6s} │ {'True':>8s} │ {'Predicted':>10s} │ {'Error':>8s}")
    print(f"   {'─'*6}─┼─{'─'*8}─┼─{'─'*10}─┼─{'─'*8}")
    for i in range(len(X_test)):
        err = abs(y_true[i, 0] - y_pred[i, 0])
        print(f"   {X_test[i,0]:6.2f} │ {y_true[i,0]:8.4f} │ {y_pred[i,0]:10.4f} │ {err:8.4f}")

    # Predict with confidence
    preds, confidence = brain.predict_with_confidence(X_test)
    print(f"\n📊 Confidence Estimates:")
    for i in range(len(X_test)):
        print(f"   x={X_test[i,0]:5.2f}: prediction={preds[i,0]:7.4f}, "
              f"confidence={confidence[i]:.3f}")


# ============================================================
# DEMO 6: Dreaming (Memory Consolidation)
# ============================================================
def demo_dreaming(brain):
    """Let the brain dream and consolidate"""
    separator("STEP 6: DREAMING", "💤")

    print("The brain enters a dream state to consolidate memories...\n")
    brain.dream(n_cycles=3, verbose=True)

    print(f"\n📊 After dreaming:")
    print(f"   Dream log entries: {len(brain.dream_log)}")
    if brain.hippocampus:
        stats = brain.hippocampus.get_memory_stats()
        print(f"   Long-term memories: {stats['long_term_size']}")
        print(f"   Consolidations: {stats['consolidations']}")


# ============================================================
# DEMO 7: Introspection
# ============================================================
def demo_introspection(brain):
    """Brain examines its own state"""
    separator("STEP 7: INTROSPECTION", "🔬")

    intro = brain.introspect()

    print("🧠 Brain State:")
    print(f"   Mode: {intro['state']['mode']}")
    print(f"   Arousal: {intro['state']['arousal']:.2f}")
    print(f"   Working Memory Items: {intro['state']['working_memory_items']}")

    print(f"\n📊 Metrics:")
    metrics = intro['metrics']
    for key, val in metrics.items():
        if isinstance(val, float):
            print(f"   {key}: {val:.6f}")
        else:
            print(f"   {key}: {val}")

    if intro['consciousness']:
        print(f"\n💭 Consciousness:")
        for key, val in intro['consciousness'].items():
            print(f"   {key}: {val}")

    if intro['memory']:
        print(f"\n💾 Memory:")
        for key, val in intro['memory'].items():
            print(f"   {key}: {val}")

    if intro['learning']:
        print(f"\n🧬 Learning:")
        for key, val in intro['learning'].items():
            print(f"   {key}: {val}")


# ============================================================
# DEMO 8: Full Summary
# ============================================================
def demo_summary(brain):
    """Print the full brain summary"""
    separator("STEP 8: BRAIN SUMMARY", "📋")
    brain.summary()


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "🧠" * 30)
    print("   PRECIOUS BRAIN — WORKING DEMO")
    print("🧠" * 30)
    print("\nThis demo walks through every capability of the brain.\n")

    start = time.time()

    # Step 1: Create
    brain = demo_create_brain()

    # Step 2: Think, Feel, Decide
    demo_cognitive_ops(brain)

    # Step 3: Memory
    demo_memory(brain)

    # Step 4: Learn
    X_train, y_train = demo_learning(brain)

    # Step 5: Predict
    demo_prediction(brain, X_train, y_train)

    # Step 6: Dream
    demo_dreaming(brain)

    # Step 7: Introspect
    demo_introspection(brain)

    # Step 8: Summary
    demo_summary(brain)

    elapsed = time.time() - start

    print(f"\n{'━'*60}")
    print(f"  ✅ ALL DEMOS COMPLETE!")
    print(f"  ⏱️  Total time: {elapsed:.1f}s")
    print(f"{'━'*60}\n")

    print("💡 What you can do next:")
    print("   • Change the training data (try y = x² or y = cos(x))")
    print("   • Increase neurons for richer thoughts")
    print("   • Add more evolution generations for better learning")
    print("   • Feed real data from files or sensors")
    print("   • Use brain.act() to execute system commands")
    print("   • Use brain.execute_command('ls -la') for shell access")
    print()

    return brain


if __name__ == "__main__":
    brain = main()
