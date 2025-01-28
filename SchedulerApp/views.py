from django.http.response import HttpResponse
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import *
from collections import defaultdict
import random
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import os
from django.conf import settings

POPULATION_SIZE = 100
NUMB_OF_ELITE_SCHEDULES = 10
TOURNAMENT_SELECTION_SIZE = 15
MUTATION_RATE = 0.01
VARS = {'generationNum': 0,
        'terminateGens': False}


fitness_values = []

class Population:
    def __init__(self, size):
        self._size = size
        # self._data = data
        self._schedules = [Schedule().initialize() for i in range(size)]

    def getSchedules(self):
        return self._schedules
    
    def __str__(self):
        # Create a string representation of the Population
        schedule_descriptions = [str(schedule) for schedule in self._schedules]
        return (f"Population Size: {self._size}\n"
                f"Schedules: \n" + "\n".join(schedule_descriptions))
    
    


class Data:
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()
        self._sections = Section.objects.all()

    def get_rooms(self):
        return self._rooms

    def get_instructors(self):
        return self._instructors

    def get_courses(self):
        return self._courses

    def get_depts(self):
        return self._depts

    def get_meetingTimes(self):
        return self._meetingTimes

    def get_sections(self):
        return self._sections


class Class:
    def __init__(self, dept, section, course):
        self.department = dept
        self.course = course
        self.instructor = None
        self.meeting_time = None
        self.room = None
        self.section = section

    def get_id(self):
        return self.section_id # see this later

    def get_dept(self):
        return self.department

    def get_course(self):
        return self.course

    def get_instructor(self):
        return self.instructor

    def get_meetingTime(self):
        return self.meeting_time

    def get_room(self):
        return self.room

    def set_instructor(self, instructor):
        self.instructor = instructor

    def set_meetingTime(self, meetingTime):
        self.meeting_time = meetingTime

    def set_room(self, room):
        self.room = room
        
    def __str__(self):
        return f"Class(dept={self.department}, course={self.course}, section={self.section}, instructor={self.instructor}, meeting_time={self.meeting_time}, room={self.room})"



class Schedule:
    def __init__(self):
        self._data = data
        self._classes = []
        self._numberOfConflicts = 0
        self._fitness = -1
        self._isFitnessChanged = True

    def getClasses(self):
        self._isFitnessChanged = True
        return self._classes

    def getNumbOfConflicts(self):
        return self._numberOfConflicts

    def getFitness(self):
        if self._isFitnessChanged:
            self._fitness = self.calculateFitness()
            self._isFitnessChanged = False
        return self._fitness

    def addCourse(self, data, course, courses, dept, section):
        newClass = Class(dept, section.section_id, course)

        newClass.set_meetingTime(
            data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])

        # newClass.set_room(
        #     data.get_rooms()[random.randrange(0, len(data.get_rooms()))])

        crs_inst = course.instructors.all()
        newClass.set_instructor(
            crs_inst[random.randrange(0, len(crs_inst))])

        self._classes.append(newClass)
    def getGenes(self):
        """Returns a list of dictionaries representing the genes (classes) in the schedule."""
        return [
            {
                "department": cls.get_dept(),
                "course": cls.get_course(),
                "instructor": cls.get_instructor(),
                "meeting_time": cls.get_meetingTime(),
                "room": cls.get_room(),
                "section": cls.section,
            }
            for cls in self._classes
        ]

    def initialize(self):
        sections = Section.objects.all()
        # print(f"Total sections found: {sections.count()}")  # Debug: Number of sections

        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            # print(f"Processing section: {section} (Department: {dept}) with {n} classes per week.")

            available_meeting_times = len(data.get_meetingTimes())
            # print(f"Available meeting times: {available_meeting_times}")

            if n > available_meeting_times:
                print(f"Reducing n from {n} to available meeting times {available_meeting_times}.")
                n = available_meeting_times  # Ensure we don't exceed available meeting times

            courses = dept.courses.all()
            # print(f"Number of courses available in department '{dept}': {courses.count()}")

            # Calculate how many classes to add
            classes_to_add = n // len(courses)
            # print(f"Adding {classes_to_add} classes for each of {courses.count()} courses.")

            for course in courses:
                for i in range(classes_to_add):
                    self.addCourse(data, course, courses, dept, section)

            additional_classes = n % len(courses)
            # print(f"Adding {additional_classes} additional classes for random courses.")

            for course in courses.order_by('?')[:additional_classes]:
                self.addCourse(data, course, courses, dept, section)

            total_classes = len(self._classes)  # Count total classes added for this section
            # print(f"Total classes added for section {section}: {total_classes}")
            


        # print(f"Finished initializing schedules. Total classes in this population: {len(self._classes)}")
        return self
    
    def parse_time(self, time_str):
        """Convert a time string (e.g., '11:30') to a time object."""
        return datetime.strptime(time_str.strip(), '%H:%M').time()


    def calculateFitness(self):
        # Dictionaries to track violations for each constraint
        self._hard_constraint_violations = {
            'same_course_same_section': 0,
            'instructor_conflict': 0,
            'duplicate_time_section': 0,
            'instructor_availability': 0,
            'total_classes_mismatch': 0,
            'course_frequency': 0  # New constraint for course frequency
        }

        self._soft_constraint_violations = {
            'no_consecutive_classes': 0,
            'noon_classes': 0,
            'break_time_conflict': 0,
            'balanced_days': 0,
        }

        # Define weights
        hard_weights = {
            'same_course_same_section': 3,
            'instructor_conflict': 3,
            'duplicate_time_section': 3,
            'instructor_availability': 3,
            'total_classes_mismatch': 3,
            'course_frequency': 5, 
        }

        soft_weights = {
            'no_consecutive_classes': 0.5,
            'noon_classes': 3,
            'break_time_conflict': 0.3,
            'balanced_days': 0.3,
        }

        # Retrieve all scheduled classes
        classes = self.getClasses()

        # Check hard constraints
        self.check_total_classes(classes, hard_weights)
        for i in range(len(classes)):
            self.check_course_conflicts(classes, i, hard_weights)
            self.check_instructor_conflict(classes, i, hard_weights)
            self.check_duplicate_time(classes, i, hard_weights)
            self.check_instructor_availability(classes, i, hard_weights)
        self.check_course_frequency(classes, hard_weights)
        # Check soft constraints
        for i in range(len(classes)):
            self.check_consecutive_classes(classes, i, soft_weights)
            self.check_noon_classes(classes, i, soft_weights)
            self.check_break_time_conflict(classes, i, soft_weights)
            
        self.check_balanced_days(classes, soft_weights)
        # Calculate penalties
        hard_penalty = sum(
            hard_weights[key] * self._hard_constraint_violations[key] for key in hard_weights
        )
        soft_penalty = sum(
            soft_weights[key] * self._soft_constraint_violations[key] for key in soft_weights
        )
        
        hard_penalty /= max(1, len(hard_weights))  # Prevent division by zero
        soft_penalty /= max(1, len(soft_weights))
        
        print(f"Hard Penalty: {hard_penalty}")  # Debug print
        print(f"Soft Penalty: {soft_penalty}")
        
        # Calculate fitness using the given formula
        total_penalty = soft_penalty + (hard_penalty ** 2)
        fitness = (1+10)/(total_penalty+1)
        print(f"Fitness: {fitness}") 
        # Assign fitness value to the schedule
        self._fitness = fitness
        return self._fitness
    
    
    def check_course_frequency(self, classes, hard_weights):
    # Create a dictionary to track the number of occurrences for each course
        course_count = {}

        for cls in classes:
            course = cls.course
            if course not in course_count:
                course_count[course] = 0
            course_count[course] += 1

        # For each course, check if it appears the required number of times per week
        for course, count in course_count.items():
            required_count = course.max_period  # assuming each course has this attribute
            if count != required_count:
                print(f"Violation: {course} should appear {required_count} times but appears {count} times.")
                self._hard_constraint_violations['course_frequency'] += 1
    
    
    def check_total_classes(self, classes, weights):
        # Initialize a dictionary to track the number of classes per section
        section_classes = {}

        # print("Checking total classes per section...")  # Debug print

        for cls in classes:
            section = cls.section  # Assuming each class has a 'section' attribute
            
            # If section is a string (like section_id), retrieve the actual Section object
            if isinstance(section, str):
                section = Section.objects.get(section_id=section)
            
            if section not in section_classes:
                section_classes[section] = 0
            section_classes[section] += 1

        # print(f"Total classes per section: {section_classes}")  # Debug print

        # Check if the number of classes in each section matches the expected number
        for section, num_classes in section_classes.items():
            # Retrieve the section's allowed number of classes (e.g., from the Section model)
            allowed_classes = section.num_class_in_week
            # print(f"Section: {section}, Total Classes: {num_classes}, Allowed Classes: {allowed_classes}")  # Debug print

            # Violation if total classes do not match the expected number
            if num_classes != allowed_classes:
                # print(f"Violation: Section {section} has a mismatch in total classes.")  # Debug print
                self._hard_constraint_violations['total_classes_mismatch'] += 1


    


    def check_course_conflicts(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            day_i = str(classes[i].meeting_time).split()[0]
            day_j = str(classes[j].meeting_time).split()[0]
            if (classes[i].course.course_name == classes[j].course.course_name and 
                day_i == day_j and classes[i].section == classes[j].section):
                self._hard_constraint_violations['same_course_same_section'] += 1

    def check_instructor_conflict(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section != classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time and 
                classes[i].instructor == classes[j].instructor):
                self._hard_constraint_violations['instructor_conflict'] += 1

    def check_duplicate_time(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if (classes[i].section == classes[j].section and 
                classes[i].meeting_time == classes[j].meeting_time):
                self._hard_constraint_violations['duplicate_time_section'] += 1

    def check_instructor_availability(self, classes, i, weights):
        instructor = classes[i].instructor
        availability_start = instructor.availability_start
        availability_end = instructor.availability_end
        meeting_time_str = classes[i].meeting_time.time
        start_time_str, end_time_str = meeting_time_str.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time = self.parse_time(end_time_str)

        if start_time < availability_start or end_time > availability_end:
            self._hard_constraint_violations['instructor_availability'] += 1

    def check_consecutive_classes(self, classes, i, weights):
        for j in range(i + 1, len(classes)):
            if classes[i].instructor == classes[j].instructor:
                time_i_end = self.parse_time(classes[i].meeting_time.time.split(' - ')[1])
                time_j_start = self.parse_time(classes[j].meeting_time.time.split(' - ')[0])
                if time_i_end == time_j_start:
                    self._soft_constraint_violations['no_consecutive_classes'] += 1

    def check_noon_classes(self, classes, i, weights):
        noon_start = self.parse_time('10:00')
        noon_end = self.parse_time('15:00')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)

        if noon_start <= start_time <= noon_end:
            self._soft_constraint_violations['noon_classes'] += 1

    def check_break_time_conflict(self, classes, i, weights):
        break_start = self.parse_time('10:00')
        break_end = self.parse_time('10:50')
        start_time_str, _ = classes[i].meeting_time.time.split(' - ')
        start_time = self.parse_time(start_time_str)
        end_time_str, _ = classes[i].meeting_time.time.split(' - ')
        end_time = self.parse_time(end_time_str)

        if start_time < break_end and end_time > break_start:
            self._soft_constraint_violations['break_time_conflict'] += 1

    def check_balanced_days(self, classes, weights):
        day_class_count = {}
        for cls in classes:
            day = str(cls.meeting_time).split()[0]
            if day not in day_class_count:
                day_class_count[day] = 0
            day_class_count[day] += 1
        max_day = max(day_class_count.values())
        min_day = min(day_class_count.values())
        if max_day - min_day > 2:
            self._soft_constraint_violations['balanced_days'] += 1



class GeneticAlgorithm:
    def __init__(self, initial_temperature=1.0, cooling_rate=0.99):
        self.temperature = initial_temperature  # Initial temperature for mutation
        self.cooling_rate = cooling_rate  # Cooling rate for temperature reduction

    def evolve(self, population):
        return self._mutatePopulation(self._crossoverPopulation(population))

    def _crossoverPopulation(self, popula):
        crossoverPopula = Population(0)
        
        # Add elite schedules directly to the new population
        for i in range(NUMB_OF_ELITE_SCHEDULES):
            crossoverPopula.getSchedules().append(popula.getSchedules()[i])

        # Perform tournament selection and crossover for the rest of the population
        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            scheduleX = self._tournamentPopulation(popula)
            scheduleY = self._tournamentPopulation(popula)
            crossoverPopula.getSchedules().append(self._crossoverSchedule(scheduleX, scheduleY))

        return crossoverPopula

    def _mutatePopulation(self, population):
        # Mutate the population, skipping elite schedules
        for i in range(NUMB_OF_ELITE_SCHEDULES, POPULATION_SIZE):
            self._mutateSchedule(population.getSchedules()[i])
        return population

    def _crossoverSchedule(self, scheduleX, scheduleY):
        # Perform one-point crossover to generate a new schedule
        crossoverSchedule = Schedule().initialize()
        for i in range(0, len(crossoverSchedule.getClasses())):
            if random.random() > 0.5:
                crossoverSchedule.getClasses()[i] = scheduleX.getClasses()[i]
            else:
                crossoverSchedule.getClasses()[i] = scheduleY.getClasses()[i]
        return crossoverSchedule


    def _mutateSchedule(self, mutateSchedule):
        schedule = Schedule().initialize()  # Create a new Schedule and initialize it
        for i in range(len(mutateSchedule.getClasses())):  # Iterate over the classes in the schedule
            if MUTATION_RATE > random.random():  # If the random value is less than the mutation rate
                mutateSchedule.getClasses()[i] = schedule.getClasses()[i]  # Replace the class at index i with a class from the new schedule
        return mutateSchedule  # Return the mutated schedule



    def _tournamentPopulation(self, popula):
        # Perform tournament selection to pick the best schedule
        tournamentPopula = Population(0)

        # Select schedules for the tournament
        for i in range(0, TOURNAMENT_SELECTION_SIZE):
            tournamentPopula.getSchedules().append(
                popula.getSchedules()[random.randrange(0, POPULATION_SIZE)])

        # Return the schedule with the best fitness from the tournament
        return max(tournamentPopula.getSchedules(), key=lambda x: x.getFitness())




def context_manager(schedule):
    classes = schedule.getClasses()
    context = []
    for i in range(len(classes)):
        clas = {}
        clas['section'] = classes[i].section_id
        clas['dept'] = classes[i].department.dept_name
        clas['course'] = f'{classes[i].course.course_name} ({classes[i].course.course_number} {classes[i].course.max_numb_students})'
        clas['room'] = f'{classes[i].room.r_number} ({classes[i].room.seating_capacity})'
        clas['instructor'] = f'{classes[i].instructor.name} ({classes[i].instructor.uid})'
        clas['meeting_time'] = [
            classes[i].meeting_time.pid,
            classes[i].meeting_time.day,
            classes[i].meeting_time.time
        ]
        context.append(clas)
    return context


def apiGenNum(request):
    return JsonResponse({'genNum': VARS['generationNum']})

def apiterminateGens(request):
    VARS['terminateGens'] = True
    return redirect('home')




from random import choice

def get_random_color():
    # Generate light colors by ensuring RGB values are higher than 200
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    
    # Return the color in the hex format
    return f"#{r:02x}{g:02x}{b:02x}"



@login_required
def timetable(request):
    global data
    data = Data()
    population = Population(POPULATION_SIZE)
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False
    populations = []  # Store all populations for metrics
    populations.append(population)  # Add initial population
    fitness_values = []  # Best fitness per generation
    average_fitness = []  # Average fitness per generation
    diversity = []  # Population diversity per generation (unique fitness values)

    geneticAlgorithm = GeneticAlgorithm()
    schedule = population.getSchedules()[0]

    while (schedule.getFitness() != 1.0) and (VARS['generationNum'] <= 150):
        if VARS['terminateGens']:
            return HttpResponse('')

        population = geneticAlgorithm.evolve(population)
        population.getSchedules().sort(key=lambda x: x.getFitness(), reverse=True)
        schedule = population.getSchedules()[0]
        populations.append(population)  # Add current population
        fitness_values.append(schedule.getFitness())  # Track best fitness

        # Calculate average fitness
        avg_fitness = sum(schedule.getFitness() for schedule in population.getSchedules()) / POPULATION_SIZE
        average_fitness.append(avg_fitness)

        # Calculate population diversity (unique fitness values)
        unique_fitness = len(set(schedule.getFitness() for schedule in population.getSchedules()))
        diversity.append(unique_fitness)

        VARS['generationNum'] += 1
        genes = schedule.getGenes()  # Assumes the getGenes() method exists in the Schedule class
        print(f'\n> Generation #{VARS["generationNum"]}, Fitness: {schedule.getFitness()}')
        print(f'Genes of Best Schedule: {genes}')

    # Generate Combined Graph
    generate_combined_plots(fitness_values, average_fitness, diversity, population_size=POPULATION_SIZE, mutation_rate=MUTATION_RATE)


    break_time_slot = '10:00 - 10:50'  # The break time you want to use
    week_days = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday']  # List of weekdays
    
    teacher_colors = {}
    instructor_names = {}

    for cls in schedule.getClasses():
        teacher = cls.get_instructor()  # Assuming this is an instructor object
        teacher_name = teacher.name  # Ensure you're getting the correct name attribute of the instructor
        
        if teacher_name not in teacher_colors:
            teacher_colors[teacher_name] = get_random_color()
        
        # Now you can store the instructor's name (or ID) for reference
        instructor_names[cls] = teacher_name

    # Generate break times for all weekdays
    break_times = [(break_time_slot, day) for day in week_days]

    return render(
        request, 'timetable.html', {
            'schedule': schedule.getClasses(),
            'sections': data.get_sections(),
            'times': data.get_meetingTimes(),
            'timeSlots': TIME_SLOTS,
            'weekDays': DAYS_OF_WEEK,
            'break_times': break_times,
            'teacher_colors': teacher_colors,
        })


def generate_combined_plots(fitness_values, average_fitness, diversity, population_size, mutation_rate):
    # Create a single figure with 3 subplots (since you only want the first 3 graphs)
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # Adjusting the layout for 3 subplots

    # Best and Average Fitness Plot
    axs[0].plot(range(len(fitness_values)), fitness_values, label='Best Fitness', color='blue', linestyle='-', marker='o')
    axs[0].plot(range(len(average_fitness)), average_fitness, label='Average Fitness', color='orange', linestyle='--', marker='x')
    axs[0].set_title(f'Best and Average Fitness Over Generations\nPopulation: {population_size}, Mutation Rate: {mutation_rate} with big tournament size')
    axs[0].set_xlabel('Generation Number')
    axs[0].set_ylabel('Fitness') 
    axs[0].grid(True)
    axs[0].legend()

    # Fitness Improvement Plot (Best - Average)
    fitness_improvement = [best - avg for best, avg in zip(fitness_values, average_fitness)]
    axs[1].plot(range(len(fitness_improvement)), fitness_improvement, label='Fitness Improvement (Best - Average)', color='purple', linestyle='-', marker='s')
    axs[1].set_title(f'Fitness Improvement Over Generations\nPopulation: {population_size}, Mutation Rate: {mutation_rate}')
    axs[1].set_xlabel('Generation Number')
    axs[1].set_ylabel('Fitness Difference')
    axs[1].grid(True)
    axs[1].legend()

    # Diversity Plot
    axs[2].plot(range(len(diversity)), diversity, label='Diversity', color='green', linestyle='-', marker='d')
    axs[2].set_title(f'Diversity Over Generations\nPopulation: {population_size}, Mutation Rate: {mutation_rate}')
    axs[2].set_xlabel('Generation Number')
    axs[2].set_ylabel('Diversity')
    axs[2].grid(True)
    axs[2].legend()

    # Adjust layout and save the combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(settings.MEDIA_ROOT, 'combined_three_100_withsbittournament_size.png'))
    plt.close()







'''
Page Views
'''

def home(request):
    return render(request, 'index.html', {})


@login_required
def instructorAdd(request):
    form = InstructorForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('instructorAdd')
    context = {'form': form}
    return render(request, 'instructorAdd.html', context)


@login_required
def instructorEdit(request):
    context = {'instructors': Instructor.objects.all()}
    return render(request, 'instructorEdit.html', context)


@login_required
def instructorDelete(request, pk):
    inst = Instructor.objects.filter(pk=pk)
    if request.method == 'POST':
        inst.delete()
        return redirect('instructorEdit')


@login_required
def roomAdd(request):
    form = RoomForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('roomAdd')
    context = {'form': form}
    return render(request, 'roomAdd.html', context)


@login_required
def roomEdit(request):
    context = {'rooms': Room.objects.all()}
    return render(request, 'roomEdit.html', context)


@login_required
def roomDelete(request, pk):
    rm = Room.objects.filter(pk=pk)
    if request.method == 'POST':
        rm.delete()
        return redirect('roomEdit')


@login_required
def meetingTimeAdd(request):
    form = MeetingTimeForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('meetingTimeAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'meetingTimeAdd.html', context)


@login_required
def meetingTimeEdit(request):
    context = {'meeting_times': MeetingTime.objects.all()}
    return render(request, 'meetingTimeEdit.html', context)


@login_required
def meetingTimeDelete(request, pk):
    mt = MeetingTime.objects.filter(pk=pk)
    if request.method == 'POST':
        mt.delete()
        return redirect('meetingTimeEdit')


@login_required
def courseAdd(request):
    form = CourseForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('courseAdd')
        else:
            print('Invalid')
    context = {'form': form}
    return render(request, 'courseAdd.html', context)


@login_required
def courseEdit(request):
    instructor = defaultdict(list)
    for course in Course.instructors.through.objects.all():
        course_number = course.course_id
        instructor_name = Instructor.objects.filter(
            id=course.instructor_id).values('name')[0]['name']
        instructor[course_number].append(instructor_name)

    context = {'courses': Course.objects.all(), 'instructor': instructor}
    return render(request, 'courseEdit.html', context)


@login_required
def courseDelete(request, pk):
    crs = Course.objects.filter(pk=pk)
    if request.method == 'POST':
        crs.delete()
        return redirect('courseEdit')


@login_required
def departmentAdd(request):
    form = DepartmentForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('departmentAdd')
    context = {'form': form}
    return render(request, 'departmentAdd.html', context)


@login_required
def departmentEdit(request):
    course = defaultdict(list)
    for dept in Department.courses.through.objects.all():
        dept_name = Department.objects.filter(
            id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(
            course_number=dept.course_id).values(
                'course_name')[0]['course_name']
        course[dept_name].append(course_name)

    context = {'departments': Department.objects.all(), 'course': course}
    return render(request, 'departmentEdit.html', context)


@login_required
def departmentDelete(request, pk):
    dept = Department.objects.filter(pk=pk)
    if request.method == 'POST':
        dept.delete()
        return redirect('departmentEdit')


@login_required
def sectionAdd(request):
    form = SectionForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            form.save()
            return redirect('sectionAdd')
    context = {'form': form}
    return render(request, 'sectionAdd.html', context)


@login_required
def sectionEdit(request):
    context = {'sections': Section.objects.all()}
    return render(request, 'sectionEdit.html', context)


@login_required
def sectionDelete(request, pk):
    sec = Section.objects.filter(pk=pk)
    if request.method == 'POST':
        sec.delete()
        return redirect('sectionEdit')




'''
Error pages
'''

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})
