(define (domain blocks-world)
  (:requirements :strips :typing)
  
  (:types block) ; Define a single type 'block'
  
  (:predicates
    (on ?x - block ?y - block)       ; Predicate to check if block ?x is on block ?y
    (ontable ?x - block)             ; Predicate to check if block ?x is on the table
    (clear ?x - block)               ; Predicate to check if block ?x has no block on top of it
    (handempty)                      ; Predicate to check if the hand is empty
    (holding ?x - block)             ; Predicate to check if the hand is holding block ?x
  )
  
  (:action pick-up
    :parameters (?x - block)
    :precondition (and (clear ?x) (ontable ?x) (handempty))
    :effect (and (not (ontable ?x)) (not (clear ?x))
                 (not (handempty)) (holding ?x))
  )

  (:action put-down
    :parameters (?x - block)
    :precondition (holding ?x)
    :effect (and (ontable ?x) (clear ?x)
                 (handempty) (not (holding ?x)))
  )
  
  (:action stack
    :parameters (?x - block ?y - block)
    :precondition (and (holding ?x) (clear ?y))
    :effect (and (not (holding ?x)) (not (clear ?y))
                 (clear ?x) (on ?x ?y) (handempty))
  )
  
  (:action unstack
    :parameters (?x - block ?y - block)
    :precondition (and (on ?x ?y) (clear ?x) (handempty))
    :effect (and (holding ?x) (clear ?y)
                 (not (on ?x ?y)) (not (handempty)))
  )
)
