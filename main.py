import numpy as np
import math
import random
import matplotlib.pyplot as plt

ALPHA = 1.0
LAMBDA = 1.0
GAMMA = 1.0
CONFIDENCE_FACTOR = 2000  # Inertia to avoid small-sample spikes

BETA_L = 1.2    # Weight for Likes
BETA_C = 1.5    # Weight for Comments
BETA_T = 1.0    # Weight for Time spent
TAU_0 = 15.0    # Reference time

# A final scalar to bring the microscopic decimal result of the exponents
# up to the 5,000 - 50,000 range so it can compete with F(V)
GLOBAL_SCALAR = 500000


def calculate_F(views, v0):
    return ALPHA * v0 * (1 - math.exp(-(views / v0) ** LAMBDA))

def calculate_S(engagement, views, v0):
    return engagement + calculate_F(views, v0)

def calculate_S_final(relevance, global_score):
    return (relevance ** GAMMA) * global_score


NUM_USERS = 150000
VECTOR_DIM = 2  # [tech, sports]

print(f"GENERATING {NUM_USERS} BACKGROUND USERS")

# Generate the random matrix of users (150,000 x 2)
user_matrix = np.random.rand(NUM_USERS, VECTOR_DIM)

# the output is a column shape (150,000 x 1) and every row is the norm of
# the embedding vector related to the row user.
user_norms = np.linalg.norm(user_matrix, axis=1, keepdims=True)

# Divide every user's vector by its own length so the length of
# every vector in the matrix to become exactly 1.0.
user_matrix = user_matrix / user_norms

class NamedUser:
    def __init__(self, name, persona, vec):
        self.name = name
        self.persona = persona
        self.vec = np.array(vec) / np.linalg.norm(vec)

named_users = [
    NamedUser("user 1", "tech", [1.0, 0.0]),
    NamedUser("user 2", "sports", [0.0, 1.0]),
    NamedUser("user 3", "mixed interests", [0.6, 0.8])
]


class Post:
    def __init__(self, post_id, title, vec, base_like, base_comment, base_time, v0):
        self.post_id = post_id
        self.title = title
        self.vec = np.array(vec) / np.linalg.norm(vec)

        self.base_like = base_like  # Inherent chance to get a like
        self.base_comment = base_comment  # Inherent chance to get a comment
        self.base_time = base_time  # Average watch time in seconds

        self.v0 = v0
        self.V = 0

        self.likes = 0
        self.comments = 0
        self.total_time = 0.0

        self.E = 0.0
        self.history_V = [0]
        self.history_S = [0]

    def global_score(self):
        return calculate_S(self.E, self.V, self.v0)

    def record_history(self):
        self.history_V.append(self.V)
        self.history_S.append(self.global_score())


def run_lifecycle_simulation(v0_value, total_views=100000, batch_size=2000):
    print(f"\nRUNNING SIMULATION FOR V0 = {v0_value}")

    # a dictionary to track which posts each user has already watched.
    user_seen_posts = {i: set() for i in range(NUM_USERS)}

    post_good_tech = Post("A", "Amazing Python Tutorial", [1.0, 0.1],
                          base_like=0.9, base_comment=0.5, base_time=40.0, v0=v0_value)

    post_bad_tech = Post("B", "Fake Clickbait Tech", [1.0, 0.1],
                         base_like=0.3, base_comment=0.01, base_time=30.0, v0=v0_value)

    post_viral_sports = Post("C", "Champions League Highlights", [0.1, 1.0],
                             base_like=0.8, base_comment=0.3, base_time=25.0, v0=v0_value)

    posts = [post_good_tech, post_bad_tech, post_viral_sports]
    # Calculate the total number of iterations (steps) the simulation loop needs to run.
    steps = total_views // batch_size

    for step in range(1, steps + 1):
        for post in posts:
            # Calculate Relevance for all users
            relevance_scores = np.dot(user_matrix, post.vec)

            # Filter out users who have already seen this specific post
            for u_id in range(NUM_USERS):
                if post.post_id in user_seen_posts[u_id]:
                    relevance_scores[u_id] = -1.0

            # Select the best users for this batch (Targeted Exploration)
            best_user_indices = np.argsort(relevance_scores)[-batch_size:][::-1]

            # Simulate interactions for these specific users
            for u_id in best_user_indices:
                relevance = relevance_scores[u_id]
                user_seen_posts[u_id].add(post.post_id)
                post.V += 1

                # Simulate Like
                if random.random() < (relevance * post.base_like):
                    post.likes += 1

                # Simulate Comment
                if random.random() < (relevance * post.base_comment):
                    post.comments += 1

                # Simulate Watch Time
                actual_time = post.base_time * relevance * random.uniform(0.8, 1.2)
                post.total_time += actual_time

            denominator = post.V + CONFIDENCE_FACTOR

            p_like = post.likes / denominator
            p_comment = post.comments / denominator
            tau = post.total_time / denominator

            # The mathematical formula exactly as written in the PDF
            raw_E = (p_like ** BETA_L) * (p_comment ** BETA_C) * ((tau / TAU_0) ** BETA_T)

            # Scale it up to match the magnitude of the Views F(V)
            post.E = raw_E * GLOBAL_SCALAR

            post.record_history()

        current_views = step * batch_size
        if current_views in [5000, 20000, 50000, 100000]:
            print(f"\n>>> MILESTONE: {current_views} Views Reached")
            print(
                f"{'Post Title':<30} | {'V':<7} | {'P_Like':<6} | {'P_Comm':<6} | {'Tau':<5} | {'E (Scaled)':<10} | {'F(V)':<8} | {'S(p)':<8}")
            print("-" * 95)
            for post in posts:
                f_v = calculate_F(post.V, v0_value)

                # Calculate raw metrics for display
                p_l = post.likes / post.V if post.V > 0 else 0
                p_c = post.comments / post.V if post.V > 0 else 0
                tau_avg = post.total_time / post.V if post.V > 0 else 0

                print(
                    f"{post.title:<30} | {post.V:<7} | {p_l:<6.2f} | {p_c:<6.2f} | {tau_avg:<5.1f} | {post.E:<10.0f} | {f_v:<8.0f} | {post.global_score():<8.0f}")

    return posts


posts_v5k = run_lifecycle_simulation(v0_value=5000, total_views=100000, batch_size=1000)
posts_v50k = run_lifecycle_simulation(v0_value=50000, total_views=100000, batch_size=1000)

print("\n" + "="*58)
choice = input("Do you want to see only the first graph (1) or both (2)?\nEnter '1' or '2': ").strip()
print("="*58 + "\n")

print("Generating graphs... (Close the plot window to see the 'For You' pages)")

def plot_scenario(ax, posts, v0_val):
    colors = {'Amazing Python Tutorial': 'blue', 'Fake Clickbait Tech': 'red', 'Champions League Highlights': 'green'}
    styles = {'Amazing Python Tutorial': '-', 'Fake Clickbait Tech': '--', 'Champions League Highlights': '-'}

    for post in posts:
        ax.plot(post.history_V, post.history_S, label=post.title,
                color=colors[post.title], linestyle=styles[post.title], linewidth=2.5)

    ax.axvline(x=v0_val, color='black', linestyle=':', linewidth=2, label=f'Exploration Boundary (V0 = {v0_val})')

    ax.set_title(f"Scenario: V0 = {v0_val}", fontsize=14)
    ax.set_xlabel("Total Views (V)", fontsize=12)
    ax.set_ylabel("Global Quality Score S(p)", fontsize=12)

    ax.set_xlim(0, 100000)
    ax.set_yscale('linear')

    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc="upper left")


if choice == '1':
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 9))
    fig.suptitle("Impact of Exploration Budget (V0) on Global Score S(p)", fontsize=18, fontweight='bold')
    plot_scenario(ax1, posts_v5k, 5000)
else:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Impact of Exploration Budget (V0) on Global Score S(p)", fontsize=18, fontweight='bold')
    plot_scenario(ax1, posts_v5k, 5000)
    plot_scenario(ax2, posts_v50k, 50000)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()