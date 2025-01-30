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
import os
from django.conf import settings
import matplotlib.pyplot as plt
from io import BytesIO
import time

POPULATION_SIZE = 100
NUMB_OF_ELITE_SCHEDULES = 10
VARS = {'generationNum': 0,
        'terminateGens': False}

fitness_values = []

class Population:
    def __init__(self, size):
        self._size = size
        self._schedules = [Schedule().initialize() for i in range(size)]

    def getSchedules(self):
        return self._schedules
    
    def __str__(self):
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
        return self.section_id

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
        self._velocity = []
        self._personal_best = None
        self._global_best = None

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
        newClass.set_meetingTime(data.get_meetingTimes()[random.randrange(0, len(data.get_meetingTimes()))])
        crs_inst = course.instructors.all()
        newClass.set_instructor(crs_inst[random.randrange(0, len(crs_inst))])
        self._classes.append(newClass)

    def getGenes(self):
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
        for section in sections:
            dept = section.department
            n = section.num_class_in_week
            available_meeting_times = len(data.get_meetingTimes())
            if n > available_meeting_times:
                n = available_meeting_times
            courses = dept.courses.all()
            classes_to_add = n // len(courses)
            for course in courses:
                for i in range(classes_to_add):
                    self.addCourse(data, course, courses, dept, section)
            additional_classes = n % len(courses)
            for course in courses.order_by('?')[:additional_classes]:
                self.addCourse(data, course, courses, dept, section)
        return self

    def parse_time(self, time_str):
        return datetime.strptime(time_str.strip(), '%H:%M').time()

    def calculateFitness(self):
        self._hard_constraint_violations = {
            'same_course_same_section': 0,
            'instructor_conflict': 0,
            'duplicate_time_section': 0,
            'instructor_availability': 0,
            'total_classes_mismatch': 0,
            'course_frequency': 0
        }

        self._soft_constraint_violations = {
            'no_consecutive_classes': 0,
            'noon_classes': 0,
            'break_time_conflict': 0,
            'balanced_days': 0,
        }

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

        classes = self.getClasses()
        self.check_total_classes(classes, hard_weights)
        for i in range(len(classes)):
            self.check_course_conflicts(classes, i, hard_weights)
            self.check_instructor_conflict(classes, i, hard_weights)
            self.check_duplicate_time(classes, i, hard_weights)
            self.check_instructor_availability(classes, i, hard_weights)
        self.check_course_frequency(classes, hard_weights)
        for i in range(len(classes)):
            self.check_consecutive_classes(classes, i, soft_weights)
            self.check_noon_classes(classes, i, soft_weights)
            self.check_break_time_conflict(classes, i, soft_weights)
        self.check_balanced_days(classes, soft_weights)

        hard_penalty = sum(hard_weights[key] * self._hard_constraint_violations[key] for key in hard_weights)
        soft_penalty = sum(soft_weights[key] * self._soft_constraint_violations[key] for key in soft_weights)
        hard_penalty /= max(1, len(hard_weights))
        soft_penalty /= max(1, len(soft_weights))
        total_penalty = soft_penalty + (hard_penalty ** 2)
        fitness = (1+10)/(total_penalty+1)
        self._fitness = fitness
        return self._fitness

    def check_course_frequency(self, classes, hard_weights):
        course_count = {}
        for cls in classes:
            course = cls.course
            if course not in course_count:
                course_count[course] = 0
            course_count[course] += 1
        for course, count in course_count.items():
            required_count = course.max_period
            if count != required_count:
                self._hard_constraint_violations['course_frequency'] += 1

    def check_total_classes(self, classes, weights):
        section_classes = {}
        for cls in classes:
            section = cls.section
            if isinstance(section, str):
                section = Section.objects.get(section_id=section)
            if section not in section_classes:
                section_classes[section] = 0
            section_classes[section] += 1
        for section, num_classes in section_classes.items():
            allowed_classes = section.num_class_in_week
            if num_classes != allowed_classes:
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

class ParticleSwarmOptimization:
    def __init__(self, population_size, inertia_weight=0.5, cognitive_coeff=2.0, social_coeff=2.0):
        self.population_size = population_size
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff

    def optimize(self, population):
        schedules = population.getSchedules()
        global_best = max(population.getSchedules(), key=lambda x: x.getFitness())
        # Initialize personal best for each schedule
        for schedule in schedules:
            schedule._personal_best = schedule
            schedule._velocity = random.uniform(0, 1)  # Velocity for the entire schedule

        for _ in range(150):  # Iterations
                for schedule in schedules:
                    # Update velocity for the schedule (not individual classes)
                    new_velocity = (
                        self.inertia_weight * schedule._velocity +
                        self.cognitive_coeff * random.random() * (schedule._personal_best.getFitness() - schedule.getFitness()) +
                        self.social_coeff * random.random() * (global_best.getFitness() - schedule.getFitness())
                    )
                    schedule._velocity = new_velocity

                # Update schedule based on velocity (e.g., mutate meeting times)
                for cls in schedule.getClasses():
                    if random.random() < abs(schedule._velocity):
                        # Randomly adjust meeting time
                        new_time = random.choice(data.get_meetingTimes())
                        cls.set_meetingTime(new_time)

                if schedule.getFitness() > schedule._personal_best.getFitness():
                    schedule._personal_best = schedule
                if schedule.getFitness() > global_best.getFitness():
                    global_best = schedule
        return global_best

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

def get_random_color():
    r = random.randint(200, 255)
    g = random.randint(200, 255)
    b = random.randint(200, 255)
    return f"#{r:02x}{g:02x}{b:02x}"

@login_required
def timetable(request):
    global data
    data = Data()
    population = Population(POPULATION_SIZE)
    VARS['generationNum'] = 0
    VARS['terminateGens'] = False

    # Performance tracking variables
    fitness_history = []  # Best fitness per generation
    hard_violations_history = []  # Hard constraint violations per generation
    soft_violations_history = []  # Soft constraint violations per generation
    execution_times = []  # Execution time per generation
    start_time = time.time()  # Start timer

    pso = ParticleSwarmOptimization(population_size=POPULATION_SIZE)
    best_schedule = None

    while (VARS['generationNum'] <= 150) and (not VARS['terminateGens']):
        # Run PSO iteration
        best_schedule = pso.optimize(population)
        
        # Track performance metrics
        fitness_history.append(best_schedule.getFitness())
        hard_violations_history.append(sum(best_schedule._hard_constraint_violations.values()))
        soft_violations_history.append(sum(best_schedule._soft_constraint_violations.values()))
        execution_times.append(time.time() - start_time)

        VARS['generationNum'] += 1
        print(f"Generation {VARS['generationNum']} - Fitness: {best_schedule.getFitness()}")

    # Generate performance graphs
    plt.figure(figsize=(15, 10))
    
    # Fitness Plot
    plt.subplot(2, 2, 1)
    plt.plot(fitness_history, color='blue', marker='o', linestyle='-')
    plt.title('Fitness Value Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)

    # Constraint Violations Plot
    plt.subplot(2, 2, 2)
    plt.plot(hard_violations_history, label='Hard Constraints', color='red')
    plt.plot(soft_violations_history, label='Soft Constraints', color='orange')
    plt.title('Constraint Violations Over Time')
    plt.xlabel('Generation')
    plt.ylabel('Violations')
    plt.legend()
    plt.grid(True)

    # Execution Time Plot
    plt.subplot(2, 2, 3)
    plt.plot(execution_times, color='green', marker='s', linestyle='--')
    plt.title('Cumulative Execution Time')
    plt.xlabel('Generation')
    plt.ylabel('Time (seconds)')
    plt.grid(True)

    # Save plots to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)

    # Prepare performance summary
    performance_summary = {
        'fitness_values': fitness_history,
        'hard_violations': hard_violations_history,
        'soft_violations': soft_violations_history,
        'execution_times': execution_times,
        'total_time': time.time() - start_time,
    }

    # Print performance summary to console
    print("\nPerformance Summary:")
    print(f"Final Fitness: {fitness_history[-1]}")
    print(f"Final Hard Violations: {hard_violations_history[-1]}")
    print(f"Final Soft Violations: {soft_violations_history[-1]}")
    print(f"Total Execution Time: {performance_summary['total_time']:.2f} seconds")

    break_time_slot = '10:00 - 10:50'  # The break time you want to use
    week_days = ['Sunday','Monday', 'Tuesday', 'Wednesday', 'Thursday']  # List of weekdays

    teacher_colors = {}
    instructor_names = {}

    for cls in best_schedule.getClasses():
            teacher = cls.get_instructor()  # Assuming this is an instructor object
            teacher_name = teacher.name  # Ensure you're getting the correct name attribute of the instructor
        
            if teacher_name not in teacher_colors:
                teacher_colors[teacher_name] = get_random_color()
        
            # Now you can store the instructor's name (or ID) for reference
            instructor_names[cls] = teacher_name

    # Generate break times for all weekdays
    break_times = [(break_time_slot, day) for day in week_days]

    return render(request, 'timetable.html', {
        'schedule': best_schedule.getClasses(),
        'sections': data.get_sections(),
        'times': data.get_meetingTimes(),
        'timeSlots': TIME_SLOTS,
        'weekDays': DAYS_OF_WEEK,
        'break_times': break_times,
        'teacher_colors': teacher_colors,
        'performance_graph': buffer.getvalue().decode('latin1'),  # Embed graph in response
    })

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
        instructor_name = Instructor.objects.filter(id=course.instructor_id).values('name')[0]['name']
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
        dept_name = Department.objects.filter(id=dept.department_id).values('dept_name')[0]['dept_name']
        course_name = Course.objects.filter(course_number=dept.course_id).values('course_name')[0]['course_name']
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

def error_404(request, exception):
    return render(request,'errors/404.html', {})

def error_500(request, *args, **argv):
    return render(request,'errors/500.html', {})