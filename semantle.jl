using Word2Vec

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