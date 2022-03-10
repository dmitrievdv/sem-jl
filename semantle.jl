using Word2Vec
using Printf

function saveword2vec(wv :: WordVectors, filename; type = :binary)
    if type == :binary
        saveword2vecbinary(wv, filename)
    end
end

function saveword2vecbinary(wv :: WordVectors, filename)
    out = open(filename*".bin", "w")
    vector_size, vocab_size = size(wv.vectors)
    write(out, vocab_size)
    write(out, vector_size)
    for i = 1:vocab_size
        write(out, wv.vocab[i]*" ")
        for j = 1:vector_size
            write(out, Float32(wv.vectors[j,i]))
        end
    end
    close(out)
end

function printwords(wv :: WordVectors, n = 5000)
    open("secretWords.js", "w") do f
        println(f, "secretWords = [")
        words = rand(wv.vocab, n)
        w = words[1]
        print(f,"\"$w\"")
        for w in words[2:end]
            print(f,",\n\"$w\"")
        end
        println(f, "\n]")
    end
end

function shufflesecret()
    words = open("secretWords.js", "r") do f
        lines = readlines(f)
        String.(getfield.(match.(r"[а-я]+", lines[2:end-1]),:match))
    end
    shuffle!(words)
    open("secretWords.js", "w") do f
        println(f, "secretWords = [")
        w = words[1]
        print(f,"\"$w\"")
        for w in words[2:end]
            print(f,",\n\"$w\"")
        end
        println(f, "\n]")
    end
end

function cleansecret(wv :: WordVectors)
    words = open("secretWords.js", "r") do f
        lines = readlines(f)
        String.(getfield.(match.(r"[а-я]+", lines[2:end-1]),:match))
    end
    checked_words = []
    for w in words
        println(w)
        if w in wv.vocab
            push!(checked_words, w)
        end
    end
    open("secretWords.js", "w") do f
        println(f, "secretWords = [")
        w = checked_words[1]
        print(f,"\"$w\"")
        for w in checked_words[2:end]
            print(f,",\n\"$w\"")
        end
        println(f, "\n]")
    end
end

function saveword2vecformat(wv :: WordVectors, filename)
    out = open(filename*".bin", "w")
    vector_size, vocab_size = size(wv.vectors)
    write(out, "$vocab_size $vector_size\n")
    for i = 1:vocab_size
        write(out, wv.vocab[i]*" ")
        for j = 1:vector_size
            write(out, Float32(wv.vectors[j,i]))
        end
    end
    close(out)
end

function loadword2vecbinary(filename)
    open(filename*".bin", "r") do f
        vocab_size, vector_size = reinterpret(Int, read(f, sizeof(Int)*2))
        vocab = Vector{String}(undef, vocab_size)
        vectors = zeros(vector_size, vocab_size)
        for i = 1:vocab_size
            vocab[i] = strip(readuntil(f, " "))
            vectors[:,i] = reinterpret(Float32, read(f, sizeof(Float32)*vector_size))
        end
        Word2Vec.WordVectors(vocab, vectors)
    end
end

function loadword2vecrussian(filename)
    open(filename*".bin", "r") do f
        vocab_size, vector_size = parse.(Int, split(readline(f)))
        vocab = Vector{String}(undef, vocab_size)
        vectors = zeros(vector_size, vocab_size)
        indeces = Int[]
        for i = 1:vocab_size
            word = split(strip(readuntil(f, " ")), '_')[1]
            vector = reinterpret(Float32, read(f, sizeof(Float32)*vector_size))
            vocab[i] = word
            vectors[:,i] = vector/√(sum(vector .^ 2))
            if occursin(r"^[а-я\-]+$", word)
                push!(indeces, i)
            end
        end
        Word2Vec.WordVectors(vocab[indeces], vectors[:, indeces])
    end
end



loadword2vec() = loadword2vecbinary("cleaned")

function loadgoogleword2vec( ; w2vdata = "GoogleNews-vectors-negative300.bin")
    wv = wordvectors(w2vdata, Float32, kind = :binary, skip=true)
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

function cleanword2vec(wv, word_list)
    n_words = length(wv.vocab)
    real_words_indices = Int[]
    real_words = readlines(word_list)
    for ind in 1:n_words
        if wv.vocab[ind] in real_words
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
    n_guesses = length(guesses)
    n_counts = min(n, n_guesses)
    worst_indices = sortperm(similarities)
    best_indices = reverse(worst_indices)
   
    rnd = rand()
    n_counts = min(n, n_guesses)
    sum_sim_best = sum(similarities[best_indices[1:n_counts]])
    sum_sim_worst = sum(1 .- similarities[worst_indices[1:n_counts]])
    
    

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
            for i in 1:n_counts
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
                ind = worst_indices[i]
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
                i = worst_indices[j]
                @printf(" (%s %.2f)", guesses[i], similarities[i])
            end
            println("")
            return from_worst_word
        else
            print("best: $from_best_word")
            for j=1:n_counts
                i = best_indices[j]
                @printf(" (%s %.2f)", guesses[i], similarities[i])
            end
            println("")
            return from_best_word
        end
    end
end

function cosinetovec(wv::WordVectors, vector, n=10)
    metrics = wv.vectors'*vector
    topn_positions = sortperm(metrics[:], rev = true)[1:n]
    topn_metrics = metrics[topn_positions]
    return topn_positions, topn_metrics
end

function semantlemathgame(wv :: WordVectors, word :: AbstractString, start_guesses; n_guesses = 10, closing = 1e3)
    guesses = deepcopy(start_guesses[1:n_guesses])
    vector_size = size(wv.vectors)[1]
    sims = zeros(n_guesses)
    last_vectors = zeros(vector_size, n_guesses)
    for i = 1:n_guesses
        w = guesses[i]
        s = similarity(wv, w, word)
        sims[i] = s
        last_vectors[:, i] = get_vector(wv, w)
    end
    best_indices = reverse(sortperm(sims))
    best_guess = guesses[best_indices[1]]
    if best_guess == word
        println("Randomly guessed!")
    end
    guess = best_guess
    vocab_size = length(wv.vocab)
    direction = zeros(vector_size)
    last_guesses = deepcopy(guesses)
    last_sims = deepcopy(sims)
    prediction_guesses = deepcopy(guesses)
    prediction_sims = deepcopy(sims)
    n_tries = n_guesses
    print(last_guesses[best_indices[1]])
    while !(word in last_guesses)
        direction .= 0.0
        sum_sim = sum(last_sims)
        mean_sim = 0.0
        best_i = best_indices[1]
        for j = 2:n_guesses
            i = best_indices[j]
            direction += (1 - last_sims[i])*(last_vectors[:, best_i] - last_vectors[:, i])/(9 - sum_sim + last_sims[best_i])
            mean_sim += (1 - last_sims[i])*last_sims[i]/(9 - sum_sim + last_sims[best_i])
        end
        direction_norm = sqrt(sum(direction .^ 2))

        Δsim = last_sims[best_i] - mean_sim
        # println("$(1- last_sims[best_i]) $Δsim $direction_norm")
        # println(direction)

        prediction_vector = last_vectors[:, best_i] + direction*(1- last_sims[best_i])/Δsim#*closing
        indx, metr = cosinetovec(wv, prediction_vector, length(guesses))
        j = 0
        for i = 1:length(guesses)
            w = wv.vocab[indx[i]]
            if w in guesses
                continue
            end
            n_tries += 1
            j += 1
            push!(guesses, w)
            sim = similarity(wv, word, w)
            push!(sims, sim)
            if w == word
                println("\nFound after $n_tries guesses!")
                return guesses, sims
            end 
            last_guesses[j] = w
            last_sims[j] = sim
            last_vectors[:, j] = get_vector(wv, w)
            if j == n_guesses
                break
            end
        end
        # last_guesses .= wv.vocab[indx]
        print(" -> $(last_guesses[1])")
        best_indices = sortperm(last_sims, rev = true)
        # append!(guesses, last_guesses)
        # append!(sims, last_sims)
        # readline()
    end
    println("")
    return guesses, sims
end

function semantlemathgame(wv :: WordVectors, word :: AbstractString, start_guess :: AbstractString; n_guesses = 10, closing = 1e3)
    start_guesses = cosine_similar_words(wv, start_guess, n_guesses)
    semantlemathgame(wv, word, start_guesses, n_guesses = n_guesses, closing = closing)
end

function semantlemathgame(wv :: WordVectors, word :: AbstractString; n_guesses = 10, closing = 1e3)
    start_guesses = rand(wv.vocab, n_guesses)
    semantlemathgame(wv, word, start_guesses, n_guesses = n_guesses, closing = closing)
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
    @printf("%13sBest guesses\n", "")
    for j=1:n
        i_best = best_ind[j]; i_worst = best_ind[end-j+1] 
        @printf("Guess #%3i %20s %6.2f\n", i_best, guesses[i_best], 100*sims[i_best])
    end
    @printf("\n\n%13sWorst guesses\n", "")
    for j=1:n
        i_best = best_ind[j]; i_worst = best_ind[end-j+1] 
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
    @printf("%13sBest guesses\n", "")
    for j=1:n
        i_best = best_ind[j]; i_worst = best_ind[end-j+1] 
        @printf("Guess #%3i %20s %6.2f\n", i_best, guesses[i_best], 100*sims[i_best])
    end
    @printf("\n\n%13sWorst guesses\n", "")
    for j=1:n
        i_best = best_ind[j]; i_worst = best_ind[end-j+1] 
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