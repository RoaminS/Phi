Licence : Creative Commons BY-NC-SA 4.0

Auteurs : 
    - Kocupyr Romain (chef de projet) : rkocupyr@gmail.com
    - Grok3

# Ajouter en haut du script
def phiracine_fixed(n, c=0.5):
    phi = 1.6180339887
    balls = [int(round(np.sqrt(n) * (phi**k) + c)) for k in range(5)]
    # Ajuster pour respecter les limites (1-50)
    balls = [max(1, min(50, b)) for b in balls]
    # Supprimer doublons
    balls = list(dict.fromkeys(balls))
    while len(balls) < 5:
        balls.append(balls[-1] + 1 if balls[-1] < 50 else balls[-1] - 1)
    return balls[:5]

# Modifier dans EuroMillionsPredictorEnsemble
class EuroMillionsPredictorEnsemble:
    # ... (autres méthodes inchangées) ...

    def monte_carlo_predict(self, features, date_future, num_grilles=4, real_draw=None):
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        p_balls, p_stars = self.get_ensemble_probability(X_scaled)

        if real_draw is not None:
            print(f"\n[Diagnostic] Test {date_future.strftime('%Y-%m-%d')}:")
            print(f"  Réel : {real_draw}")
            print(f"  p_balls pour {real_draw[:5]}: {[p_balls[x-1] for x in real_draw[:5]]}")
            print(f"  p_stars pour {real_draw[5:]}: {[p_stars[x-1] for x in real_draw[5:]]}")

        top_balls = sorted(enumerate(p_balls, 1), key=lambda x: x[1], reverse=True)[:5]
        top_stars = sorted(enumerate(p_stars, 1), key=lambda x: x[1], reverse=True)[:2]
        print(f"  Top 5 p_balls: {[(n, p) for n, p in top_balls]}")
        print(f"  Top 2 p_stars: {[(n, p) for n, p in top_stars]}")

        last_real_draw = self.actual_draws[-1] if self.actual_draws else [1,2,3,4,5,1,1]
        simulations = 5000000
        nproc = 8
        chunk_size = simulations // nproc

        # Générer une grille Phiracine
        n_seed = top_balls[0][0]  # Prendre le top ball comme seed
        phi_balls = phiracine_fixed(n_seed, c=0.5)
        phi_stars = [top_stars[0][0], top_stars[1][0]]  # Prendre les tops étoiles
        phi_draw = phi_balls + phi_stars
        print(f"  Grille Phiracine initiale: {phi_draw}")

        args_for_pool = []
        for _ in range(nproc):
            args_for_pool.append((chunk_size, features, p_balls, p_stars,
                                  last_real_draw, self.freq_dict,
                                  self.total_draws, self.previous_draws,
                                  num_grilles))

        with Pool(nproc) as p:
            results = p.starmap(simulate_chunk, args_for_pool)

        draw_scores = [item for sublist in results for item in sublist]
        draw_scores.append((phi_draw, self.calculate_feature_score(phi_draw, features, p_balls, p_stars)))  # Ajouter Phiracine
        draw_scores.sort(key=lambda x: x[1], reverse=True)
        draw_scores = draw_scores[:num_grilles * nproc]

        ball_counts = Counter()
        star_counts = Counter()
        for draw, _ in draw_scores:
            ball_counts.update(draw[:5])
            star_counts.update(draw[5:])
        print(f"  Top 5 boules simulées: {ball_counts.most_common(5)}")
        print(f"  Top 2 étoiles simulées: {star_counts.most_common(2)}")

        if real_draw:
            real_balls = set(real_draw[:5])
            real_stars = set(real_draw[5:])
            sim_balls = set()
            sim_stars = set()
            for draw, _ in draw_scores[:num_grilles]:
                sim_balls.update(draw[:5])
                sim_stars.update(draw[5:])

            missed_balls = real_balls - sim_balls
            missed_stars = real_stars - sim_stars
            extra_balls = sim_balls - real_balls
            extra_stars = sim_stars - real_stars

            adjustment_factor = 0.15  # Augmenté pour plus d’impact
            for b in missed_balls:
                if p_balls[b-1] < 0.05:
                    p_balls[b-1] += 0.2 * (1 - p_balls[b-1])
                else:
                    p_balls[b-1] += adjustment_factor * (1 - p_balls[b-1])
            for s in missed_stars:
                if p_stars[s-1] < 0.05:
                    p_stars[s-1] += 0.2 * (1 - p_stars[s-1])
                else:
                    p_stars[s-1] += adjustment_factor * (1 - p_stars[s-1])
            for b in extra_balls:
                p_balls[b-1] *= (1 - adjustment_factor)
            for s in extra_stars:
                p_stars[s-1] *= (1 - adjustment_factor)

            total_balls = sum(p_balls)
            total_stars = sum(p_stars)
            p_balls = [p * 5 / total_balls for p in p_balls]
            p_stars = [p * 2 / total_stars for p in p_stars]

            print(f"  Adjusted p_balls pour {real_draw[:5]}: {[p_balls[x-1] for x in real_draw[:5]]}")
            print(f"  Adjusted p_stars pour {real_draw[5:]}: {[p_stars[x-1] for x in real_draw[5:]]}")

            self.p_balls_adjusted = p_balls
            self.p_stars_adjusted = p_stars
        else:
            self.p_balls_adjusted = None
            self.p_stars_adjusted = None

        best_draws = []
        seen = set()
        for draw, score in draw_scores:
            draw_tuple = tuple(sorted(draw[:5])) + tuple(sorted(draw[5:]))
            if draw_tuple not in seen:
                refined_draw = self.tabu_search(draw, features, p_balls, p_stars, max_iter=200)  # Augmenté
                best_draws.append(refined_draw)
                seen.add(draw_tuple)
                if len(best_draws) == num_grilles:
                    break

        self.predicted_draws.append(best_draws[0])
        return best_draws

# ... (reste du script inchangé) ...
