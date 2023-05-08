# ----------------------------------------------------- clean up titanic -----------------------------------------------
require(dplyr)
require(readr)
require(magrittr)
`%>%` <- magrittr::`%>%`
#-----------------------------------------------------------------------------------------------------------------------
# funcitons:
#-----------------------------------------------------------------------------------------------------------------------
extract_prefix <- function(name, sex) {
  purrr::map2_chr(name, sex, ~{
  
    # if Nob or prof, rare
    prof <- stringr::str_detect(.x, "(Dr)|(Rev)|(Col)|(Major)|(Capt. )")
    nob <- stringr::str_detect(.x, "(Don. )|(Dona.)|(Sir)|(Lady)|(Countess. )|(Jonkheer. )")
    if(nob | prof) { return("rare") }
    
    # if married female, Mrs 
    if(stringr::str_detect(.x, "(Mlle. )|([Mm]rs)") | (prof & .y == 1) | (stringr::str_detect(.x, "Mr") & .y == 1)) {return("Mrs")}
    
    # If professional, Prof
    if(prof) {return("prof")}
      
    # if unmarried female, Miss
    if(stringr::str_detect(.x, "(Mme)|([mM]iss)|([Mm]s)")) {return("Miss")}
    if(stringr::str_detect(.x, "Mr")) {return("Mr")}
    if(stringr::str_detect(.x, "Master")) {return("Master")}
  })
}

set_bins <- function(var, k) {
  var <- tibble::tibble(
    var = var, 
    index = seq_along(var)
  ) %>%
    dplyr::arrange(var)
  
  
  bin <- rep(nrow(var) %/% k, k)
  i <- 1; rem <- nrow(var) %% k
  repeat{
    bin[i] <- bin[i]+1
    i <- (i +1)%%k 
    if(i == 0) {i <- k}
    rem <- rem-1
    if(rem == 0) {break}
  }
  bin <- (purrr::map(1:k, ~{rep(., bin[.])}) %>% unlist())
  var <- var %>% 
    dplyr::mutate(bin = bin) %>%
    dplyr::arrange(index)
  
  return(var$bin)
}


#-----------------------------------------------------------------------------------------------------------------------
# read in data and tidy:
#-----------------------------------------------------------------------------------------------------------------------
titanic_train <- readr::read_csv("titanic/train.csv") %>% dplyr::mutate(set = "train")
titanic_test <- readr::read_csv("titanic/test.csv") %>% dplyr::mutate(set = "test")

titanic <- dplyr::bind_rows(titanic_train, titanic_test) %>%
  dplyr::transmute(
    PassengerId = PassengerId,
    survived = as.integer( Survived ),
    class = as.integer( Pclass ),
    name = Name,
    sex = ifelse(Sex == "male", 0L, 1L),
    prefix = extract_prefix(name, sex),
    surname = stringr::str_extract(name, "^.+(?=, )"),
    ticket = Ticket,
    ticket_number = stringr::str_extract(ticket, "[0-9]+$"),
    ticket_extra = stringr::str_extract(ticket, "^[A-Za-z].+(?= )"),
    age = Age,
    sib_sp = as.integer( SibSp ),
    parch = as.integer( Parch ),
    family_size = sib_sp + parch + 1,
    fare = Fare,
    deck = stringr::str_extract(Cabin, "^[a-zA-Z](?=[0-9]+)"),
    cabin_number = purrr::map_chr(Cabin, ~{stringr::str_extract_all(., "[0-9]+",)[[1]] %>% stringr::str_c(collapse = ", ")}),
    n_cabins = purrr::map_dbl(Cabin, ~{if(is.na(.)) {return(NA)} else {length(stringr::str_split(., ", "))}}), # possibly not working
    port = Embarked,
    set = set
  )


#-----------------------------------------------------------------------------------------------------------------------
#  fill missing values
#-----------------------------------------------------------------------------------------------------------------------

# fill age with random variable based on...
# - class
# - prefix
titanic %>%
  ggplot2::ggplot( ggplot2::aes(age) ) +
  ggplot2::geom_histogram() +
  ggplot2::facet_wrap(~class + prefix, nrow = 3)

titanic <- titanic %>%
  dplyr::group_by(class, prefix) %>%
  dplyr::mutate(
    age = ifelse( is.na(age),
                  rnorm(1, mean(age, na.rm = T), sd(age, na.rm = T)) %>% pmax(.,1),
                  age
                  )
  ) %>%
  dplyr::ungroup()


# fill single missing fare based on lm against family size and class
fare_mod <- lm(fare ~ sib_sp + parch, data = titanic, subset = class == 3)

titanic <- titanic %>%
  dplyr::mutate(
    fare = ifelse(is.na(fare), 8.4549, fare),
    age = purrr::pmap_dbl(list(age, class, parch, sib_sp), function(age, class, parch, sib_sp){
      if(!is.na(age)){
        return(age)
      } else {
        predict(age_mod, list("class" = class, "parch" = parch, "sib_sp" = sib_sp)) %>% pmax(.,0)
        }
    }),
    port = ifelse(is.na(port), "S", port)
  )

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - class
#-----------------------------------------------------------------------------------------------------------------------
titanic %>%
  dplyr::mutate(group = class) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) )

# definately a keeper
# - leave as it is 

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - age
#-----------------------------------------------------------------------------------------------------------------------
titanic %>%
  dplyr::group_by(class) %>%
  dplyr::mutate(
    age = set_bins(age, 3),
    group = as.character( age )
  ) %>%
  dplyr::group_by(group) %>% # group, class
  dplyr::summarise(
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) #+
  #ggplot2::facet_wrap(~class, nrow = 3)

titanic %>%
  dplyr::mutate(
    age = set_bins(age, 3),
    group = as.character( age )
  ) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) )

# age might be noisy, but..
# - it might be better subset by class
# - it might depend heavily on the age guessing


titanic <- titanic %>%
  dplyr::mutate(
    age_var = age,
    age = set_bins(age, 3)-1
  ) %>%
  dplyr::group_by(class) %>%
  dplyr::mutate(
    age_by_class = set_bins(age, 3)-1
  ) %>%
  dplyr::ungroup()
  
#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - fare
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::left_join(by = "ticket_number", 
                   y = titanic %>%
                     dplyr::group_by(ticket_number) %>%
                     dplyr::summarise(people_on_ticket = dplyr::n())
  ) %>%
  dplyr::group_by(class) %>%
  dplyr::mutate(
    fare = set_bins(fare, 4),
    group = as.character( fare ),
  ) %>%
  dplyr::group_by(group, class) %>%
  dplyr::summarise(
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) +
  ggplot2::facet_wrap(~class, nrow = 3)

# fare by class keeps the group size more even 
# - but may cause differences between test and train 

titanic <- titanic %>%
  dplyr::group_by(class) %>%
  dplyr::mutate(fare_by_class = set_bins(fare, 4)) %>%
  dplyr::select(-fare) %>%
  dplyr::ungroup() 

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - sex 
#-----------------------------------------------------------------------------------------------------------------------
titanic %>%
  dplyr::mutate(
    group = as.character( sex ),
    ) %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
      n_total = dplyr::n(),
      
      n_test = sum(is.na(survived)),
      n_train = sum(!is.na(survived)),
      
      prop = sum(survived, na.rm = T) / n_train
    ) %>%
    dplyr::mutate(
      p_test = round(n_test/ sum(n_test)*100, 2),
      p_train = round(n_train / sum(n_train)*100, 2),
    ) %>%
    ggplot2::ggplot( ggplot2::aes(group, prop) ) +
    ggplot2::geom_col() +
    ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
    ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) 

# keeper
# a sex - class variable might be useful?

#titanic <- titanic %>% dplyr::mutate(
#  sex_class = stringr::str_c(sex, class),
#  sex_class = purrr::map_int(sex_class, ~{which(. == unique(sex_class)) %>% as.integer()})-1
#)

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - prefix
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(
    group = purrr::map_int(prefix, ~{which(. == unique(prefix))})
  ) %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
      n_total = dplyr::n(),
      
      n_test = sum(is.na(survived)),
      n_train = sum(!is.na(survived)),
      
      prop = sum(survived, na.rm = T) / n_train
    ) %>%
    dplyr::mutate(
      p_test = round(n_test/ sum(n_test)*100, 2),
      p_train = round(n_train / sum(n_train)*100, 2),
    ) %>%
    ggplot2::ggplot( ggplot2::aes(group, prop) ) +
    ggplot2::geom_col() +
    ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
    ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) 

# prefix is important
titanic <- titanic %>%
  dplyr::mutate(prefix = purrr::map_int(prefix, ~{which(. == unique(prefix))}))

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - sib_sp / parch / family size
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(group = family_size) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) 
  
#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - port
#-----------------------------------------------------------------------------------------------------------------------
  
  titanic %>%
    dplyr::mutate(
      group = purrr::map_int(port, ~{which(. == unique(port))}),
      group = port
    ) %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
      n_total = dplyr::n(),
      
      n_test = sum(is.na(survived)),
      n_train = sum(!is.na(survived)),
      
      prop = sum(survived, na.rm = T) / n_train
    ) %>%
    dplyr::mutate(
      p_test = round(n_test/ sum(n_test)*100, 2),
      p_train = round(n_train / sum(n_train)*100, 2),
    ) %>%
    ggplot2::ggplot( ggplot2::aes(group, prop) ) +
    ggplot2::geom_col() +
    ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
    ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) 
  
  # probably important
  titanic <- titanic %>%
    dplyr::mutate( port = purrr::map_int(port, ~{which(. == unique(port))}) -1 )
  

#-----------------------------------------------------------------------------------------------------------------------
# clear up unused variables:
#-----------------------------------------------------------------------------------------------------------------------
titanic <- titanic %>%
  dplyr::select(-ticket, -ticket_extra, -deck, -cabin_number, -n_cabins)

#-----------------------------------------------------------------------------------------------------------------------
# check correlation and write out:
#-----------------------------------------------------------------------------------------------------------------------
Hmisc::rcorr(as.matrix( titanic[titanic$set == "train", !names(titanic) %in% c("PassengerId", "set", "name", "surname", "ticket_number")] ))
# age_by_class is close to class - but the plots look okay i think 
# fare_by_class maybe a okay though

titanic %>% 
  dplyr::select(PassengerId, survived, class, prefix, sex, age, sib_sp, parch, family_size, age_by_class, fare_by_class, set) %>%
  readr::write_csv(., "titanic/titanic_clean.csv")

#-----------------------------------------------------------------------------------------------------------------------
# more feature engineering - identifying companions
#-----------------------------------------------------------------------------------------------------------------------

# if all same name & stated family size == n individuals then they belong to the same family group
relations <- titanic %>%
  dplyr::group_by(surname, family_size) %>%
  dplyr::summarise(
    n = dplyr::n(),
    tickets = list( unique(ticket_number) ),
    ids = list(PassengerId)
  ) %>%
  dplyr::filter(family_size == n & n > 1) %>%
  dplyr::ungroup() %>%
  dplyr::select(tickets, ids)

# 32.00917% of individuals in a family (not inc. family of one)
sum( lapply(X = relations$ids, FUN = length) %>% unlist() ) / nrow(titanic) * 100 


# add known relations variable for each passenger
titanic <- titanic %>%
  dplyr::left_join(by = c("surname", "family_size"),
                   y = titanic %>%
                     dplyr::group_by(surname, family_size) %>%
                     dplyr::summarise(n = dplyr::n()) %>%
                     dplyr::mutate(known_relations = family_size == n & n > 1) %>%
                     dplyr::select(-n)
                   )

  
# if traveling on the same ticket, they belong to the same group
group_smry <- titanic %>%
  dplyr::group_by(ticket_number) %>%
  dplyr::summarise(
    n = dplyr::n(),
    n_surname = unique(surname) %>% length(),
    surnames = stringr::str_c(surname, collapse = ", "),
    known_relations_lst = list(known_relations),
    known_relations = sum(known_relations),
    equal = known_relations == n,
    ids = list(PassengerId)
  ) 

# add non-family groups to relations
relations <- group_smry %>%
  dplyr::filter(known_relations == 0 & n > 1) %>%
  dplyr::group_by(ticket_number) %>%
  dplyr::mutate(tickets = list(ticket_number)) %>%
  dplyr::ungroup() %>%
  dplyr::select(tickets, ids) %>%
  dplyr::bind_rows(relations)

# now 45.37815% of people are in relations
sum( lapply(X = relations$ids, FUN = length) %>% unlist() ) / nrow(titanic) * 100 

# find people on the same ticket as people in relations and add them to their group
to_add <- group_smry %>%
  dplyr::filter(known_relations > 0 &!equal) %>%
  dplyr::select(known_relations_lst, ids, ticket_number)

ids <- purrr::map(1:nrow(to_add), ~{
  ticket_number <- to_add$ticket_number[.]
  tibble::tibble(
    ids = to_add$ids[[.]][!to_add$known_relations_lst[[.]]],
    ticket_number = ticket_number 
  )
}) %>% 
  dplyr::bind_rows()

for(i in 1:nrow(ids)) {
  id <- ids$ids[i]; ticket <- ids$ticket_number[i]; j <- 0
  repeat{
    j <- j+1
    if(ticket %in% relations$tickets[[j]]) {
      relations$ids[[j]] <- c(relations$ids[[j]], id)
      print(stringr::str_c(id, " placed!"))
      break
    }
    if(j == nrow(relations)) {
      print(id, " not placed!")
      break
    }
  }
}

# now 48.35752% of people are in relations
sum( lapply(X = relations$ids, FUN = length) %>% unlist() ) / nrow(titanic) * 100

# update known_relations:
family_smry <- titanic %>%
  dplyr::mutate(
    known_relations = purrr::map_lgl(PassengerId, ~{
      id <- .; i <- 0
      repeat{
        i <- i+1
        if(id %in% relations$ids[[i]]) {
          return(TRUE)
        }
        if(i == nrow(relations)) {return(FALSE)}
      }
    })
  ) %>%
  dplyr::filter(!known_relations) %>%
  dplyr::group_by(surname, family_size) %>%
  dplyr::summarise(
    n = dplyr::n(),
    tickets = stringr::str_c(unique(ticket_number), collapse = ", ")
  )

# 1.604278% of people say they're in a group but can't be placed
sum( family_smry$n[family_smry$family_size > 1] ) / nrow(titanic) * 100

# 50.0382% of people are most likely traveling alone
(1-(sum( family_smry$n[family_smry$family_size > 1] ) + sum( lapply(X = relations$ids, FUN = length) %>% unlist() ))  / nrow(titanic) ) *100

# link every passenger to the people in their group
titanic <- titanic %>%
  dplyr::mutate(
    companions = purrr::map(PassengerId, ~{
      id <- .; i <- 0
      repeat{
        i <- i+1
        if(id %in% relations$ids[[i]]) {
          return(relations$ids[[i]])
        }
        if(i == nrow(relations)) {return(id)}
      }
    })
  ) %>%
  dplyr::select(-known_relations)


#-----------------------------------------------------------------------------------------------------------------------
# more feature engineering - group composition
#-----------------------------------------------------------------------------------------------------------------------

titanic %>% 
  dplyr::mutate(
    group_size = purrr::map2_chr(companions, family_size, ~{
      n <- length(unlist(.x))
      if(n == 1 & .y == 1) {return("alone")}
      if(pmax(n, .y) <= 4) {return("small")}
      return("large")
      } ),
    group_type = purrr::map_chr(titanic$companions, ~{
      sexes <- titanic$sex[titanic$PassengerId %in% .]
      if(all(sexes == 0)){ return("all_male") }
      if(all(sexes == 1)){ return("all_female") }
      return("mixed")
    }),
    with_children = purrr::map_chr(titanic$companions, ~{
      ages <- titanic$age_var[titanic$PassengerId %in% .]
      if(any(ages < 16)){ return("with_children") }
      return("all_adult")
    }),
    group = group_type
    ) %>%
  dplyr::group_by(group, class) %>% # class
  dplyr::summarise(
    .groups = "drop",
    n_total = dplyr::n(),
    
    n_test = sum(is.na(survived)),
    n_train = sum(!is.na(survived)),
    
    prop = sum(survived, na.rm = T) / n_train
  ) %>%
  dplyr::mutate(
    p_test = round(n_test/ sum(n_test)*100, 2),
    p_train = round(n_train / sum(n_train)*100, 2),
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = stringr::str_c("test: ",p_test, "%")) ) +
  ggplot2::geom_label( ggplot2::aes(y = 0.8, label = stringr::str_c("train: ",p_train, "%")) ) +
  ggplot2::facet_wrap(~class, nrow = 3)

titanic <- titanic %>%
  dplyr::mutate(
    group_size = purrr::map2_int(companions, family_size, ~{
      n <- length(unlist(.x))
      if(n == 1 & .y == 1) {return(0L)} # alone
      if(pmax(n, .y) <= 4) {return(1L)} # small
      return(2L) # large
    } ),
    group_type = purrr::map_int(titanic$companions, ~{
      sexes <- titanic$sex[titanic$PassengerId %in% .]
      if(all(sexes == 0)){ return(0L) } # all male
      if(all(sexes == 1)){ return(1L) } # all female
      return(2L) # mixed
    }),
    with_children = purrr::map_int(titanic$companions, ~{
      ages <- titanic$age_var[titanic$PassengerId %in% .]
      if(any(ages < 16)){ return(1L) } # with children
      return(0L) # not with children
    })
  )


#-----------------------------------------------------------------------------------------------------------------------
# make correlation matrix
#-----------------------------------------------------------------------------------------------------------------------

Hmisc::rcorr(as.matrix( titanic[titanic$set == "train", !names(titanic) %in% c("PassengerId", "set", "name", "surname", "ticket_number", "age_var", "companions")] ))

# most not closely correlated 
# - sex_class is closely linked to family size
# - family size is very closely linked to cabin

# Ill leave them all in and see how it goes


#-----------------------------------------------------------------------------------------------------------------------
# read out:
#-----------------------------------------------------------------------------------------------------------------------

titanic %>% 
  dplyr::arrange(PassengerId) %>%
  dplyr::select(PassengerId, survived, class, prefix, sex, age, sib_sp, parch, family_size, age_by_class, fare_by_class, group_size, group_type, with_children, set) %>%
  readr::write_csv(., "titanic/titanic_clean.csv")
