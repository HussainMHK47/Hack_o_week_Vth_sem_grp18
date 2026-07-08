const express = require('express');

const app = express();

app.use(express.json());

let students = [
    {
        id: 1,
        name: "Hussain",
        age: 19,
        course: "CSE"
    },
    {
        id: 2,
        name: "Ali",
        age: 20,
        course: "AI"
    }
];

// GET all students
app.get('/students', (req, res) => {
    res.json(students);
});

// GET student by ID
app.get('/students/:id', (req, res) => {

    const id = parseInt(req.params.id);

    const student = students.find(s => s.id === id);

    if (student) {
        res.json(student);
    } else {
        res.status(404).json({
            message: "Student not found"
        });
    }

});

// POST student
app.post('/students', (req, res) => {

    const student = req.body;

    students.push(student);

    res.json({
        message: "Student Added Successfully",
        student: student
    });

});

// PUT student
app.put('/students/:id', (req, res) => {

    const id = parseInt(req.params.id);

    const student = students.find(s => s.id === id);

    if (student) {

        student.name = req.body.name;
        student.age = req.body.age;
        student.course = req.body.course;

        res.json({
            message: "Student Updated Successfully",
            student: student
        });

    } else {

        res.status(404).json({
            message: "Student not found"
        });

    }

});

// DELETE student
app.delete('/students/:id', (req, res) => {

    const id = parseInt(req.params.id);

    const index = students.findIndex(s => s.id === id);

    if (index !== -1) {

        students.splice(index, 1);

        res.json({
            message: "Student Deleted Successfully"
        });

    } else {

        res.status(404).json({
            message: "Student not found"
        });

    }

});

// Start server
app.listen(3000, () => {
    console.log("Server is running on http://localhost:3000");
});