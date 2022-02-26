using Word2Vec
using Printf

function loadcleanword2vec()
    wv = wordvectors("GoogleNews-vectors-negative300.bin", Float32, kind = :binary, skip=true)
    n_words = length(wv.vocab)
    real_words_indices = Int[]
    for ind in 1:n_words
        if occursin(r"^[a-z]+$", wv.vocab[ind])
            push!(real_words_indices, ind)
        end
    end
    println(length(real_words_indices))
    vocab = wv.vocab[real_words_indices]
    vectors = wv.vectors[:, real_words_indices]
    return Word2Vec.WordVectors(vocab, vectors)
end

function findoppositepairs(wv :: WordVectors, threshold :: Real; n = 10)
    n_words = length(wv.vocab)
    opposite_pairs = []
    for i = 1:n_words
        println(i/n_words)
        for j = 1:n_words
            if similarity(wv, wv.vocab[i], wv.vocab[j]) < threshold
                push!(opposite_pairs, (wv.vocab[i], wv.vocab[j]))
            end
            if length(opposite_pairs) == n
                break
            end
        end
        if length(opposite_pairs) == n
            break
        end
    end
    opposite_pairs
end

function wordscore(wv :: WordVectors, word; n = 10)
    similar_words = cosine_similar_words(wv, word, n)
    score = 0.0
    sum_sim = 0.0
    for w in similar_words
        sim = similarity(wv, w, word)
        similar_to_similar = cosine_similar_words(wv, word, n)
        mean_sim = 0.0
        for ww in similar_to_similar
            sim_sim = similarity(wv, ww, word)
            mean_sim += sim_sim/n
        end
        score += sim*mean_sim
        sum_sim += sim
    end
    return score/sum_sim
end

function semantleguess(wv, guesses, similarities; n = 10, rnd_prob = 0.3, worst_rare = 1.0)
    best_indices = sortperm(similarities)
    n_guesses = length(guesses)
    start_guess = if length(guesses) > n
        n_guesses - n + 1
    else
        1
    end
    rnd = rand()
    n_counts = min(n, n_guesses)
    sum_sim_best = sum(similarities[best_indices[start_guess:n_guesses]])
    sum_sim_worst = sum(1 .- similarities[best_indices[1:n_counts]])
    
    if rnd < rnd_prob
        w = guesses[1]
        while w in guesses
            w = rand(wv.vocab)
        end
        println("rand: $w")
        return w
    else
        from_best_word = ""
        from_worst_word = ""
        max_mean_sim = -1.0
        min_mean_sim = 1.0
        for w in wv.vocab
            if w in guesses
                continue
            end
            mean_sim = 0.0
            for i in start_guess:n_guesses
                ind = best_indices[i]
                g = guesses[ind]
                s = similarities[ind]
                sim = similarity(wv, w, g)
                mean_sim += sim*s/sum_sim_best
            end
            if mean_sim > max_mean_sim
                max_mean_sim = mean_sim
                from_best_word = w
            end
            mean_sim = 0.0
            for i in 1:n_counts
                ind = best_indices[i]
                g = guesses[ind]
                s = similarities[ind]
                sim = similarity(wv, w, g)
                mean_sim += sim*(1-s)/sum_sim_worst
            end
            if mean_sim < min_mean_sim
                min_mean_sim = mean_sim
                from_worst_word = w
            end
        end
        if rnd < rnd_prob + (1-rnd_prob)*(1 - sum_sim_best/n_counts)/worst_rare
            print("worst: $from_worst_word ")
            for j=1:n_counts
                i = best_indices[j]
                @printf(" (%s %.2f)", guesses[i], similarities[i])
            end
            println("")
            return from_worst_word
        else
            print("best: $from_best_word")
            for j=start_guess:n_guesses
                i = best_indices[j]
                @printf(" (%s %.2f)", guesses[i], similarities[i])
            end
            println("")
            return from_best_word
        end
    end
end

function semantlegame(wv :: WordVectors, word :: AbstractString; rnd_prob = 0.3, worst_rare = 1.0)
    guesses = rand(wv.vocab, 5)
    sims = zeros(5)
    for i = 1:5
        w = guesses[i]
        s = similarity(wv, w, word)
        sims[i] = s
    end
    if word in guesses
        println("Randomly guessed!")
        return guesses, sims
    end
    n_gueses = 5
    while guesses[end] != word
        n_gueses += 1
        print("Guess #$n_gueses from ")
        guess = semantleguess(wv, guesses, sims, rnd_prob = rnd_prob, worst_rare = worst_rare)
        s = similarity(wv, guess, word)
        println("  similarity score $(100*s)")
        push!(guesses, guess); push!(sims, s)
    end
    println("Found after $n_gueses guesses!")
    return guesses, sims
end

function printbest(n, guesses, sims)
    best_ind = reverse(sortperm(sims))
    @printf("Answer: %s. Found after %i guesses\n\n", guesses[end], length(guesses))
    @printf("%13sBest guesses%13s %17sWorst guesses%13s\n", "", "", "", "")
    for j=1:n
        i_best = best_ind[j+1]; i_worst = best_ind[end-j+1] 
        @printf("Guess #%3i %20s %6.2f     ", i_best, guesses[i_best], 100*sims[i_best])
        @printf("Guess #%3i %20s %6.2f\n", i_worst, guesses[i_worst], 100*sims[i_worst])
    end
end

function printbest(n, wv, word :: AbstractString, guesses)
    n_g = length(guesses)
    sims = zeros(n_g)
    for i = 1:n_g   
        sims[i] = similarity(wv, word, guesses[i])
    end
    best_ind = reverse(sortperm(sims))
    @printf("Guesses relative to %s. Answer %s found after found after %i guesses\n\n", word, guesses[end], length(guesses))
    @printf("%13sBest guesses%13s %17sWorst guesses%13s\n", "", "", "", "")
    for j=1:n
        i_best = best_ind[j]; i_worst = best_ind[end-j+1] 
        @printf("Guess #%3i %20s %6.2f     ", i_best, guesses[i_best], 100*sims[i_best])
        @printf("Guess #%3i %20s %6.2f\n", i_worst, guesses[i_worst], 100*sims[i_worst])
    end
end

function printsimilar(wv :: WordVectors, word :: AbstractString; n = 10)
    words = cosine_similar_words(wv, word, n)
    for i = 1:n
        s = similarity(wv, word, words[i])
        @printf("%20s %6.2f\n", words[i], 100*s)
    end
end