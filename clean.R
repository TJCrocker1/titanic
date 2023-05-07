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
    #prefix <- stringr::str_extract_all(., "(Mme)|(Mlle. )|([mM]iss)|([Mm]rs)|([Mm]r)|([Mm]aster)|(Dr)|(Rev)|(Col)|(Major)|(Capt. )|(Don. )|(Sir)|(Lady)|(Countess. )|(Jonkheer. )")

    # if Nob or prof, rare
    prof <- stringr::str_detect(.x, "(Dr)|(Rev)|(Col)|(Major)|(Capt. )")
    nob <- stringr::str_detect(.x, "(Don. )|(Dona.)|(Sir)|(Lady)|(Countess. )|(Jonkheer. )")
    if(nob | prof) { return("rare") }
    
    # if married female, Mrs 
    # (inc professional females - these are likely to be assuming their husbands title because sexism and all that, right?)
    if(stringr::str_detect(.x, "(Mlle. )|([Mm]rs)") | (prof & .y == 1) | (stringr::str_detect(.x, "Mr") & .y == 1)) {return("Mrs")}
    
    # If professional, Prof
    if(prof) {return("prof")}
      
    # if unmarried female, Miss
    if(stringr::str_detect(.x, "(Mme)|([mM]iss)|([Mm]s)")) {return("Miss")}
    
    if(stringr::str_detect(.x, "Mr")) {return("Mr")}
    if(stringr::str_detect(.x, "Master")) {return("Master")}
    #if(stringr::str_detect(.x, "[Mm]s")) {return("Ms")}
  })
}

#extract_prefix(titanic$name, titanic$sex)
# Mr, Mrs, Miss, Master, Prof, Nob


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

# Age: ----------------------------------------------------------------------
# age might be important but there are 86 NAs in age for the test data 

# observations: ---------------------------------------------------------------
# - class has some relationship to age (the rich are older) but fare does not.
# - sib_sp and parch (i.e. family size) are both negatively related to age.
# - lower decks with lower classes appear to be younger but lots of missing data (maybe means no cabin assigned?)
# - there are no NAs in class sib_sp or parch

# solution: ------------------------------------------------------------------
# - for the test data set estimate missing ages from class, sib_sp and parch
# - for the training data remove missing rows (?)

# Fare: ---------------------------------------------------------------------
# - fare might be important but there there is a single missing fare value in test

# observations: ---------------------------------------------------------------
# - the missing value is class 3, no family group, 60.5yrs old and unknown deck
# - there is a strong relationship between class and fare, especially for upper class
# - higher fairs for more central decks (but widely dispersed unknowns)
# - some relationship between sib_sp and parch, especially for lowest class
# - some relationship between embarkation port and fare, S is most expensive in lower class
# - little relationship between age or sex and fare

# solution: -------------------------------------------------------------------
# - Estimate the missing fare from sib_sp, parch and embarkation port
# - ignore class 2 and 3 in the estimation

# embarked: (port) -------------------------------------------------------------------
# just one missing, fill with "S" the most common

fare_mod <- lm(fare ~ sib_sp + parch, data = titanic, subset = class == 3)
age_mod <- lm(age ~ factor(class) + sib_sp + parch, data = titanic) 

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
  dplyr::group_by(class) %>%
  dplyr::summarise(n = sum(survived == 1, na.rm = T)/dplyr::n()) %>%
  ggplot2::ggplot( ggplot2::aes(class, n) ) +
  ggplot2::geom_col()

# definately a keeper
# - leave as it is 

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - sex (& class)
#-----------------------------------------------------------------------------------------------------------------------
titanic %>%
  dplyr::group_by(sex, class) %>%
  dplyr::summarise(prop = sum(survived, na.rm = T) / dplyr::n()) %>%
  ggplot2::ggplot( ggplot2::aes(stringr::str_c(sex, "_", class), prop) ) +
  ggplot2::geom_col()

# keeper
# a sex - class variable might be useful?

titanic <- titanic %>% dplyr::mutate(
  sex_class = stringr::str_c(sex, class),
  sex_class = purrr::map_int(sex_class, ~{which(. == unique(sex_class)) %>% as.integer()})-1
)

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - prefix
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::group_by(prefix) %>%
  dplyr::summarise(prop = sum(survived, na.rm = T) / dplyr::n()) %>%
  ggplot2::ggplot( ggplot2::aes(prefix, prop) ) +
  ggplot2::geom_col()

# super useful 
# - leave as it is

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - sib_sp / parch / family size
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(family_size = sib_sp + parch + 1) %>%
  dplyr::group_by(sib_sp) %>%
  dplyr::summarise(prop = sum(survived, na.rm = T) / dplyr::n()) %>%
  ggplot2::ggplot( ggplot2::aes(sib_sp, prop) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 10, 1))

titanic %>%
  dplyr::mutate(family_size = sib_sp + parch + 1) %>%
  dplyr::group_by(parch) %>%
  dplyr::summarise(prop = sum(survived, na.rm = T) / dplyr::n()) %>%
  ggplot2::ggplot( ggplot2::aes(parch, prop) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 10, 1))

titanic %>%
  dplyr::mutate(family_size = sib_sp + parch + 1) %>%
  dplyr::group_by(family_size) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / sum(!is.na(survived))
    ) %>%
  ggplot2::ggplot( ggplot2::aes(family_size, prop, alpha = n) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 10, 1))

# family size may be more useful than sib_sp and parch
# - add family size and remove the others

titanic <- titanic %>%
  dplyr::mutate(family_size = sib_sp + parch + 1) %>%
  dplyr::select(-c(sib_sp, parch))
  
  

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - surname
#-----------------------------------------------------------------------------------------------------------------------

family_smry <- titanic %>%
  dplyr::group_by(surname, family_size) %>%
  dplyr::summarise(
    n = dplyr::n(),
    ticket_groups = unique(ticket_number) %>% length(),
    tickets = stringr::str_c(ticket_number, collapse = ", ")
  ) %>%
  dplyr::mutate(family_known = family_size == n) %>%
  dplyr::filter(family_known) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(family_id = seq_along(surname)) %>%
  dplyr::select(surname, family_size, family_id, n)

sum(family_smry$n) / nrow(titanic) #  0.8036669

# family can be identified for ~80\% of individuals. However it is not clear what other features could be extracted

titanic <- titanic %>% dplyr::select(-name, -surname)

#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - ticket number
#-----------------------------------------------------------------------------------------------------------------------

group_smry <- titanic %>%
  dplyr::group_by(ticket_number) %>%
  dplyr::summarise(
    n = dplyr::n(),
    #n_surname = unique(surname) %>% length(),
    #surnames = stringr::str_c(surname, collapse = ", ")
  ) %>%
  dplyr::ungroup() %>%
  dplyr::mutate( group_id = seq_along(ticket_number) ) %>%
  dplyr::select(ticket_number, group_id, "group_size" = n)


titanic %>%
  dplyr::left_join(group_smry, by = c("ticket_number")) %>%
  dplyr::group_by(group_size) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / sum(!is.na(survived))
  )  %>%
  ggplot2::ggplot( ggplot2::aes(group_size, prop, alpha = n) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 10, 1))

# group size looks similar to family size but it might be useful for mixed groups

titanic <- titanic %>%
  dplyr::left_join(group_smry, by = c("ticket_number")) %>%
  dplyr::select(-ticket, -ticket_number, -ticket_extra, -group_id)


#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - age
#-----------------------------------------------------------------------------------------------------------------------


titanic %>%
  dplyr::mutate(age = set_bins(age, 4)) %>%
  dplyr::mutate(
    #group = age,
    group = stringr::str_c("c", class, ", a", age)
    #group = stringr::str_c("s", sex, ", a", age),
    #group = stringr::str_c( ", a", age,"p", prefix)
    ) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / n
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = n) )

# age isn't much good by itself, but..
# - combined with class its much better
# - combined with sex its kinda okay but not awesome
# - combined with prefix its also sorta fine but not ideal


titanic <- titanic %>%
  dplyr::mutate(
    age = set_bins(age, 4),
    class_age = stringr::str_c(class, age),
    class_age = purrr::map_int(class_age, ~{which(. == (unique(class_age)%>%sort())) %>% as.integer()})-1
  )


#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - fare
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(
    fare = set_bins(fare, 4),
    fare_group = set_bins(fare/group_size, 4),
    fare_family = set_bins(fare/family_size, 4),
    
    #group = as.character( fare ),
    #group = stringr::str_c("c", class, ", f", fare),
    #group = stringr::str_c("f", fare, ", a", age ),
    #group = stringr::str_c("f", fare, ", ", prefix),
    
    #group = as.character( fare_group ),
    #group = stringr::str_c("c", class, ", f", fare_group),
    #group = stringr::str_c("f", fare_group, ", a", age),
    #group = stringr::str_c(prefix, ", f", fare_group),
    
    group = as.character( fare_family ),
    group = stringr::str_c("c", class, ", f", fare_family),
    #group = stringr::str_c("f", fare, ", a", age),
    #group = stringr::str_c(prefix, ", f", fare_family)

  ) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / n
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = n) )

# fair looks good
# - fair by age  group size is even, but doesn't add much beyond fare
# - fair by class has very variable group size
# - fair by prefix has variable group size

# fair_group looks bad - no variation
# - maybe slight intra-class vatiation, higher fairs lower survival ??
# - age by fair group has no variation
# - by prefix dosn't seem to add anything

# fair by family - minimal variation probably nothing
# - class may have a strong interaction with fare within class 2
# - age doesn't add anything really
# - mot sure about prefix

titanic <- titanic %>%
  dplyr::mutate(
    fare = set_bins(fare, 4) -1,
    fare_family = set_bins(fare/family_size, 4),
    class_fare_family = stringr::str_c(class, fare_family),
    class_fare_family = purrr::map_int(class_fare_family, ~{which(. == (unique(class_fare_family) %>% sort())) %>% as.integer()})-1
  ) %>%
  dplyr::select(-fare_family)
  
#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - deck
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(
    group = purrr::map_int(deck, ~{
      if(is.na(.)) {return(0L)}
      which(. == unique(deck[!is.na(deck)]))
      }),
    group = ifelse(is.na(deck), 0L, 1L)
    ) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / n
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = n) )

titanic$n_cabins %>% unique()

# deck doesn't seem to have much influence, and most are NA.
# - if you had a cabin you are more likely to survive even if it's a bad one.
# - lots of cabin numbers, not sure if there's anything useful there
# - n_cabins is broken

titanic <- titanic %>%
  dplyr::mutate( cabin = ifelse(is.na(deck), 0L, 1L) ) %>%
  dplyr::select(-deck, -cabin_number, -n_cabins)


#-----------------------------------------------------------------------------------------------------------------------
# feature engineering - prefix
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(
    group = purrr::map_int(prefix, ~{which(. == unique(prefix))})
  ) %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / n
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = n) )

# prefix is very important, should probably be one-hot encoded rather than ordinal

titanic <- titanic %>%
  dplyr::mutate(value = 1L) %>%
  tidyr::spread(prefix, value, fill = 0L) %>%
  dplyr::select(-rare)

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
    n = sum(!is.na(survived)),
    prop = sum(survived, na.rm = T) / n
  ) %>%
  ggplot2::ggplot( ggplot2::aes(group, prop) ) +
  ggplot2::geom_col() +
  ggplot2::geom_label( ggplot2::aes(y = 0.9, label = n) )

titanic <- titanic %>%
  dplyr::mutate(
    prefix = extract_prefix(name, sex),
    family_size = sib_sp + parch,
     fare_bracket = purrr::map_chr(fare, function(fare) {
      if(fare < 15) return("Fare_lowest")
      if(fare < 50) return("Fare_low")
      if(fare < 100) return("Fare_med")
      if(fare < 200) return("Fare_high")
      return("Fare_highest")
    }),
    age_bracket = purrr::map_chr(age, function(age) {
      if(age < 12) return("child")
      if(age < 20) return("teenager")
      if(age < 40) return("adult")
      return("old")
    })
  ) %>%
  dplyr::select(PassengerId, survived, prefix, class, sex, age, fare, deck, embarked, family_size, fare_bracket, age_bracket, set)


#-----------------------------------------------------------------------------------------------------------------------
# possible feature engineering approaches:
#-----------------------------------------------------------------------------------------------------------------------

titanic %>%
  dplyr::mutate(fare = round(titanic$fare) ) %>%
  dplyr::arrange(fare) %>%
  dplyr::group_by(fare) %>%
  dplyr::summarise(
    n = dplyr::n()
  ) %>%
  ggplot2::ggplot( ggplot2::aes( fare, n ) ) +
  ggplot2::geom_col() +
  ggplot2::scale_x_continuous(breaks = seq(0, 500, 25)) +
  ggplot2::coord_cartesian(xlim = c(0, 50))

stringr::str_extract(titanic$ticket, "[0-9]+")

length(unique(titanic$ticket))

# family size:
# parch + sib_sp

# age structure

# adult female with dependents: sex = 1 & age >= 20 & parch > 0
# adult male with dependants: sex = 0 & age >= 20 & parch > 0

# titles from names




#-----------------------------------------------------------------------------------------------------------------------
# read out:
#-----------------------------------------------------------------------------------------------------------------------
readr::write_csv(titanic, "titanic/titanic_clean.csv")

# 0.8518518518518519