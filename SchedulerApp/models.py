from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.contrib.auth.models import AbstractUser
from django.db.models.signals import post_save, post_delete


TIME_SLOTS = (
    ('6:45 - 8:20' , '6:45 - 8:20'),
    ('8:20 - 10:00'  , '8:20 - 10:00'),
    ('10:00 - 10:50', '10:00 - 10:50'),
    ('10:50 - 12:30', '10:50 - 12:30'),
    ('12:30 - 2:10'  , '12:30 - 2:10'),
)


DAYS_OF_WEEK = (
    ('Sunday', 'Sunday'),
    ('Monday', 'Monday'),
    ('Tuesday', 'Tuesday'),
    ('Wednesday', 'Wednesday'),
    ('Thursday', 'Thursday'),

)
    # ('Saturday', 'Saturday'),


class Room(models.Model):
    r_number = models.CharField(max_length=6)
    seating_capacity = models.IntegerField(default=0)

    def __str__(self):
        return self.r_number


class Instructor(models.Model):
    uid = models.CharField(max_length=6)
    name = models.CharField(max_length=25)
    availability_start = models.TimeField(null=True)  # Start time in 24-hour format
    availability_end = models.TimeField(null=True)    # End time in 24-hour format

    def __str__(self):
        return f'{self.uid} {self.name} (Available: {self.availability_start} - {self.availability_end})'


class MeetingTime(models.Model):
    pid = models.CharField(max_length=4, primary_key=True)
    time = models.CharField(max_length=50,
                            choices=TIME_SLOTS,
                            default='11:30 - 12:30')
    day = models.CharField(max_length=15, choices=DAYS_OF_WEEK)

    def __str__(self):
        return f'{self.pid} {self.day} {self.time}'
    
class InstructorAvailability(models.Model):
    instructor = models.ForeignKey(Instructor, on_delete=models.CASCADE, related_name='availability')
    meeting_time = models.ForeignKey(MeetingTime, on_delete=models.CASCADE, related_name='available_instructors')

    def __str__(self):
        return f'{self.instructor} is available at {self.meeting_time}'


class Course(models.Model):
    course_number = models.CharField(max_length=5, primary_key=True)
    course_name = models.CharField(max_length=40)
    max_period = models.IntegerField(default=2)
    instructors = models.ManyToManyField(Instructor)

    def __str__(self):
        return f'{self.course_number} {self.course_name}'


class Department(models.Model):
    dept_name = models.CharField(max_length=50)
    courses = models.ManyToManyField(Course)

    @property
    def get_courses(self):
        return self.courses

    def __str__(self):
        return self.dept_name


class Section(models.Model):
    section_id = models.CharField(max_length=25, primary_key=True)
    department = models.ForeignKey(Department, on_delete=models.CASCADE)
    num_class_in_week = models.IntegerField(default=0)
    course = models.ForeignKey(Course,
                               on_delete=models.CASCADE,
                               blank=True,
                               null=True)
    meeting_time = models.ForeignKey(MeetingTime,
                                     on_delete=models.CASCADE,
                                     blank=True,
                                     null=True)
    room = models.ForeignKey(Room,
                             on_delete=models.CASCADE,
                             blank=True,
                             null=True)
    instructor = models.ForeignKey(Instructor,
                                   on_delete=models.CASCADE,
                                   blank=True,
                                   null=True)

    def set_room(self, room):
        section = Section.objects.get(pk=self.section_id)
        section.room = room
        section.save()

    def set_meetingTime(self, meetingTime):
        section = Section.objects.get(pk=self.section_id)
        section.meeting_time = meetingTime
        section.save()

    def set_instructor(self, instructor):
        section = Section.objects.get(pk=self.section_id)
        section.instructor = instructor
        section.save()


'''
class Data(models.Manager):
    def __init__(self):
        self._rooms = Room.objects.all()
        self._meetingTimes = MeetingTime.objects.all()
        self._instructors = Instructor.objects.all()
        self._courses = Course.objects.all()
        self._depts = Department.objects.all()

    def get_rooms(self): return self._rooms

    def get_instructors(self): return self._instructors

    def get_courses(self): return self._courses

    def get_depts(self): return self._depts

    def get_meetingTimes(self): return self._meetingTimes

    def get_numberOfClasses(self): return self._numberOfClasses

'''
