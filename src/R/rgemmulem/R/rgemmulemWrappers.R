# wrapper functions
rgemmulem_expectationmaximization <- function(compats, counts) {
    result <- .Call("r_expectationmaximization", compats, counts);
    return (result);
}
rgemmulem_unmixgaussians <- function(values, num_dist) {
    result <- .Call("r_unmixgaussians", values, num_dist);
    return (result);
}
rgemmulem_unmixexponentials <- function(values, num_dist) {
    result <- .Call("r_unmixexponentials", values, num_dist);
    return (result);
}
